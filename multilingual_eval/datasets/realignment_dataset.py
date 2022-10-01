# Chinese load_dataset("un_multi", "en-zh", cache_dir=cache_dir)
# Others news_commentary
from collections import defaultdict
import itertools
import logging
from transformers import DataCollatorWithPadding
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple, List
import torch
from datasets import interleave_datasets
from datasets.iterable_dataset import IterableDataset
from torch.utils.data import DataLoader
import numpy as np

from multilingual_eval.data import get_dicos
from multilingual_eval.datasets.data_utils import (
    TorchCompatibleIterableDataset,
    convert_dataset_to_iterable_dataset,
    get_signature_columns_if_needed,
    infinite_iterable_dataset,
    repeat_iterable_dataset,
)
from multilingual_eval.datasets.translation_dataset import get_news_commentary, get_opus100
from multilingual_eval.datasets.xtreme_udpos import get_xtreme_udpos
from multilingual_eval.utils import get_tokenizer_type, subwordlist_to_wordlist


@dataclass
class BilingualDictionary:
    """
    Class for holding pairs of words in a bilingual dictionary
    """

    forward: Dict[Tuple[str], Set[Tuple[str]]]
    backward: Dict[Tuple[str], Set[Tuple[str]]]

    def sample_dictionary(self, fraction):
        """
        sample a dictionary, usefull for train/test splits
        """
        new_forward = defaultdict(lambda: set())
        new_backward = defaultdict(lambda: set())

        remaining_forward = defaultdict(lambda: set())
        remaining_backward = defaultdict(lambda: set())

        for key, values in self.forward.items():
            for value in values:
                if np.random.random() < fraction:
                    new_forward[key].add(value)
                    new_backward[value].add(key)
                else:
                    remaining_forward[key].add(value)
                    remaining_backward[value].add(key)

        self.forward = new_forward
        self.backward = new_backward

        return BilingualDictionary(remaining_forward, remaining_backward)


class DatasetMapperForRealignment:
    """
    Class for defining a callable that can be used as an argument of datasets.Dataset.map()
    Applied to a translation dataset, it will create an alignment table between tokens of
    the translated sentences and output samples for training a realignment task as expected
    by models created with multilingual_eval.models.with_realignment_factory.model_with_realignment_factory
    """

    def __init__(
        self,
        tokenizer,
        left_key: str,
        right_key: str,
        dico_path=None,
        dico=None,
        dictionary_fraction=1.0,
        split="all",
        max_length=None,
        ignore_identical=True,
        add_identical=False,
    ):
        self.tokenizer = tokenizer
        self._tokenizer_type = get_tokenizer_type(tokenizer)

        if dico is None:
            forward, backward = get_dicos(
                left_key,
                right_key,
                dico_path,
                ignore_identical=ignore_identical,
                tokenizer=tokenizer,
                split=split,
            )

            self.dico = BilingualDictionary(forward, backward)
        else:
            self.dico = dico

        if dictionary_fraction < 1.0:
            self.other_dico = self.dico.sample_dictionary(dictionary_fraction)
            forward = self.dico.forward
            backward = self.dico.backward
        else:
            self.other_dico = self.dico

        self.left_key = left_key
        self.right_key = right_key

        self._max_left_length = max(map(len, forward))
        self._max_right_length = max(map(len, backward))

        self.perfs = defaultdict(lambda: [0.0, 0])
        self.add_identical = add_identical

        self.max_length = max_length

    def tokenize_sentences(self, left_sent, right_sent) -> Tuple[List[str], List[str]]:
        """
        Tokenize both translated sentences and apply
        truncation if needed
        """

        if left_sent is None or right_sent is None:
            raise Exception(
                "null sentence found in dataset. Please filter translation datasets for null sentences"
            )

        left_tokens = self.tokenizer.tokenize(left_sent)
        right_tokens = self.tokenizer.tokenize(right_sent)

        if self.max_length is not None:
            left_tokens = left_tokens[: self.max_length - 2]
            right_tokens = right_tokens[: self.max_length - 2]

        return left_tokens, right_tokens

    def create_multi_subwords(self, left_tokens: List[str], right_tokens: List[str]):
        """
        Generates candidate multi-tokens expression for the alignment table, based
        on the maximum length in term of tokens of words in the bilingual dictionary

        Returns:
        - left_multi: a list of tuple of strings representing the multi-word
            candidate expressions in the left sentence (generally of size 1 except for chinese)
        - left_multi_pos: a list of list of int of size Nx2 indicating the start and end offset of
            each element of left_multi
        - right_multi: left_multi_pos but for the right sentence
        - right_multi_pos: left_multi_pos but bfor the right sentence
        """

        left_multi: List[Tuple[str]] = []
        left_multi_pos: List[List[int]] = []
        for length in range(1, self._max_left_length + 1):
            for i in range(0, len(left_tokens) - length + 1):
                left_multi.append(tuple(left_tokens[i : i + length]))
                left_multi_pos.append([i + 1, i + 1 + length])

        right_multi = []
        right_multi_pos = []
        for length in range(1, self._max_right_length + 1):
            for i in range(0, len(right_tokens) - length + 1):
                right_multi.append(tuple(right_tokens[i : i + length]))
                right_multi_pos.append([i + 1, i + 1 + length])

        return left_multi, left_multi_pos, right_multi, right_multi_pos

    def find_aligned_words(self, left_multi, left_multi_pos, right_multi, right_multi_pos):
        """
        Compare subsets of consecutive tokens from both sentence and add them to the alignment table
        if they are in the bilingual dictionary (and if there is no ambiguity)
        Returns:
        - aligned_left_multi_pos: a list of list of int of size Nx2 indicating the start and end offset
            (in term of subwords) of each aligned element
        - aligned_right_multi_pos: offsets of the translation of each word referenced by aligned_left_multi_pos
        """
        aligned_left_multi_pos = []
        aligned_right_multi_pos = []
        for left_pos, word in zip(left_multi_pos, left_multi):
            candidates = self.dico.forward.get(word, set()).intersection(right_multi)

            # Verify that there is one and only one candidate in set of words
            if len(candidates) != 1:
                continue

            # Verify that there is only one occurence of this word and extract it
            i = -1
            for i, (right_pos, tgt_word) in enumerate(
                filter(lambda x: x[1] in candidates, zip(right_multi_pos, right_multi))
            ):
                if i == 1:
                    break
            if i != 0:
                continue

            # Verify that the target word is not the translation of another word in the source
            backward_candidates = self.dico.backward.get(tgt_word, set())
            counter = 0
            for w in left_multi:
                if w in backward_candidates:
                    counter += 1
            if counter != 1:
                continue

            aligned_left_multi_pos.append(left_pos)
            aligned_right_multi_pos.append(right_pos)
        return aligned_left_multi_pos, aligned_right_multi_pos

    def add_identical_expressions(self, left_multi, left_multi_pos, right_multi, right_multi_pos):
        """
        Add identical expression found in both sentences
        """
        new_aligned_left_multi_pos = []
        new_aligned_right_multi_pos = []
        for (left_start, left_end), left_expression in zip(left_multi_pos, left_multi):
            for (right_start, right_end), right_expression in zip(right_multi_pos, right_multi):
                if left_expression == right_expression:
                    new_aligned_left_multi_pos.append([left_start, left_end])
                    new_aligned_right_multi_pos.append([right_start, right_end])
        return new_aligned_left_multi_pos, new_aligned_right_multi_pos

    def remove_overlapping_aligned(self, aligned_left_multi_pos, aligned_right_multi_pos):
        """
        Remove overlapping aligned pairs to avoid any confusion
        """
        to_remove = set()
        for i, (start_i, end_i) in enumerate(aligned_left_multi_pos):
            for j, (start_j, end_j) in enumerate(aligned_left_multi_pos):
                if j == i:
                    continue
                if start_i <= start_j and end_j <= end_i:
                    to_remove.add(j)
                elif start_j <= start_i and end_i <= end_j:
                    to_remove.add(j)
        for i, (start_i, end_i) in enumerate(aligned_right_multi_pos):
            for j, (start_j, end_j) in enumerate(aligned_right_multi_pos):
                if j == i:
                    continue
                if start_i <= start_j and end_j <= end_i:
                    to_remove.add(j)
                elif start_j <= start_i and end_i <= end_j:
                    to_remove.add(j)
        to_remove = sorted(list(to_remove), reverse=True)
        for i in to_remove:
            del aligned_left_multi_pos[i], aligned_right_multi_pos[i]

    def create_words_from_subwords(self, left_tokens, right_tokens):
        """
        Returns the position range of each word in each sentence in term
        of subword/token
        """

        _, left_word_positions = subwordlist_to_wordlist(
            left_tokens, split_type=self._tokenizer_type
        )
        _, right_word_positions = subwordlist_to_wordlist(
            right_tokens, split_type=self._tokenizer_type
        )
        return left_word_positions, right_word_positions

    def create_final_subword_list(
        self,
        left_word_positions,
        right_word_positions,
        aligned_left_multi_pos,
        aligned_right_multi_pos,
    ):
        """
        From the position ranges of aligned expressions (aligned_left_multi_pos and aligned_right_multi_pos)
        and from the position ranges of words (left_word_positions and right_word_positions) build lists of
        positions of all word (aligned and non-aligned) without any overlapping (alignment_left_positions and
        alignment_right_positions) as well as the index of aligned words/expressions in those list
        (alignment_left_ids and alignment_right_ids)
        """
        # Create set of positions included in aligned expressions
        left_covered_ids = set()
        for start, end in aligned_left_multi_pos:
            left_covered_ids = left_covered_ids.union(range(start, end))

        right_covered_ids = set()
        for start, end in aligned_right_multi_pos:
            right_covered_ids = right_covered_ids.union(range(start, end))

        alignment_left_ids = []
        alignment_right_ids = []
        alignment_left_positions = []
        alignment_right_positions = []

        # Add position of entire words that are not aligned nor overlapping an aligned expression
        for start, end in left_word_positions:
            if start not in left_covered_ids and end - 1 not in left_covered_ids:
                alignment_left_positions.append([start, end])

        # Build the positions of all words as the concatenation of all entire words which do not
        # overlap with aligned words, and the position of aligned words
        alignment_left_ids = list(
            range(
                len(alignment_left_positions),
                len(alignment_left_positions) + len(aligned_left_multi_pos),
            )
        )
        alignment_left_positions += aligned_left_multi_pos

        # Repeat the same for the right sentence
        for start, end in right_word_positions:
            if start not in right_covered_ids and end - 1 not in right_covered_ids:
                alignment_right_positions.append([start, end])

        alignment_right_ids = list(
            range(
                len(alignment_right_positions),
                len(alignment_right_positions) + len(aligned_right_multi_pos),
            )
        )
        alignment_right_positions += aligned_right_multi_pos

        return (
            alignment_left_ids,
            alignment_right_ids,
            alignment_left_positions,
            alignment_right_positions,
        )

    def __call__(self, example):
        """
        Take an translation sample of the form {"name_of_lang_1": "sentence", "name_of_lang_2": "sentence"}
        and return a sample for a realignment task, with properties:
        - left_*: (like left_input_ids) result of the tokenizer for the left sentence
        - right_*: same for the right sentence
        - alignment_left_positions: position range of all words in the left sentence (in term of subword)
        - alignment_right_positions: same for the right sentence
        - alignment_left_ids: index of aligned word in alignment_left_positions
        - alignment_right_ids: index of corresponding aligned words in alignment_right_positions
        - alignment_nb: the number of aligned pair (usefull for truncation)
        - alignment_left_length: the number of word in alignment_left_positions (usefull for truncation)
        - alignment_right_length: the same for the right sentence
        """
        left_sent = example[self.left_key]
        right_sent = example[self.right_key]

        left_tokens, right_tokens = self.tokenize_sentences(left_sent, right_sent)

        left_multi, left_multi_pos, right_multi, right_multi_pos = self.create_multi_subwords(
            left_tokens, right_tokens
        )

        aligned_left_multi_pos, aligned_right_multi_pos = self.find_aligned_words(
            left_multi, left_multi_pos, right_multi, right_multi_pos
        )

        if self.add_identical:
            (
                new_aligned_left_multi_pos,
                new_aligned_right_multi_pos,
            ) = self.add_identical_expressions(
                left_multi, left_multi_pos, right_multi, right_multi_pos
            )
            aligned_left_multi_pos += new_aligned_left_multi_pos
            aligned_right_multi_pos += new_aligned_right_multi_pos

        # re-merge "words" afterwards
        # first, deduplicate aligned pairs that overlap
        # (e.g. remove ["maison", "house"] if we have ["maisons", "houses"] at the same place)
        # we only deal with simple case (inclusion) no weird overlap
        self.remove_overlapping_aligned(aligned_left_multi_pos, aligned_right_multi_pos)

        # then, we merge subwords into words when possible
        left_word_positions, right_word_positions = self.create_words_from_subwords(
            left_tokens, right_tokens
        )

        (
            alignment_left_ids,
            alignment_right_ids,
            alignment_left_positions,
            alignment_right_positions,
        ) = self.create_final_subword_list(
            left_word_positions,
            right_word_positions,
            aligned_left_multi_pos,
            aligned_right_multi_pos,
        )

        return {
            **{
                f"left_{k}": v[: self.max_length] if self.max_length is not None else v
                for k, v in self.tokenizer(left_sent).items()
            },
            **{
                f"right_{k}": v[: self.max_length] if self.max_length is not None else v
                for k, v in self.tokenizer(right_sent).items()
            },
            "alignment_left_ids": alignment_left_ids,
            "alignment_right_ids": alignment_right_ids,
            "alignment_left_positions": alignment_left_positions,
            "alignment_right_positions": alignment_right_positions,
            "alignment_nb": len(alignment_left_ids),
            "alignment_left_length": len(alignment_left_positions),
            "alignment_right_length": len(alignment_right_positions),
        }


class DatasetMapperForInjectingRealignmentData:
    """
    deprecated: not useful with the new training loop
    Class for defining a callable that can be used as an argument of datasets.Dataset.map()
    Inject a realignment example inside the sample of a given task
    """

    def __init__(self, realignment_dataset):
        self.realignment_dataset = realignment_dataset
        self.realignment_iterator = iter(self.realignment_dataset)

    def __call__(self, example):
        try:
            realignment_example = next(self.realignment_iterator)
        except StopIteration:
            self.realignment_iterator = iter(self.realignment_dataset)
            realignment_example = next(self.realignment_iterator)

        return {**example, **realignment_example}


class RealignmentCollator:
    """
    Data collator for building and padding batch for the realignment task
    """

    def __init__(self, tokenizer, **kwargs):
        self.usual_collator = DataCollatorWithPadding(tokenizer, **kwargs)

    def __call__(self, examples):
        left_inputs = [
            {k.split("_", 1)[1]: v for k, v in sample.items() if k.startswith("left_")}
            for sample in examples
        ]
        right_inputs = [
            {k.split("_", 1)[1]: v for k, v in sample.items() if k.startswith("right_")}
            for sample in examples
        ]
        batch_left = {f"left_{k}": v for k, v in self.usual_collator(left_inputs).items()}
        batch_right = {f"right_{k}": v for k, v in self.usual_collator(right_inputs).items()}

        max_nb = max(map(lambda x: x["alignment_nb"], examples))
        max_left_length = max(map(lambda x: x["alignment_left_length"], examples))
        max_right_length = max(map(lambda x: x["alignment_right_length"], examples))

        alignment_left_ids = torch.zeros((len(examples), max_nb), dtype=torch.long)
        alignment_right_ids = torch.zeros((len(examples), max_nb), dtype=torch.long)
        alignment_left_positions = torch.zeros(
            (len(examples), max_left_length, 2), dtype=torch.long
        )
        alignment_right_positions = torch.zeros(
            (len(examples), max_right_length, 2), dtype=torch.long
        )

        for i, ex in enumerate(examples):
            alignment_left_ids[i, : ex["alignment_nb"]] = torch.LongTensor(ex["alignment_left_ids"])
            alignment_right_ids[i, : ex["alignment_nb"]] = torch.LongTensor(
                ex["alignment_right_ids"]
            )
            alignment_left_positions[i, : ex["alignment_left_length"]] = torch.LongTensor(
                ex["alignment_left_positions"]
            )
            alignment_right_positions[i, : ex["alignment_right_length"]] = torch.LongTensor(
                ex["alignment_right_positions"]
            )

        return {
            **batch_left,
            **batch_right,
            "alignment_left_ids": alignment_left_ids,
            "alignment_right_ids": alignment_right_ids,
            "alignment_left_positions": alignment_left_positions,
            "alignment_right_positions": alignment_right_positions,
            "alignment_nb": torch.LongTensor([ex["alignment_nb"] for ex in examples]),
            "alignment_left_length": torch.LongTensor(
                [ex["alignment_left_length"] for ex in examples]
            ),
            "alignment_right_length": torch.LongTensor(
                [ex["alignment_right_length"] for ex in examples]
            ),
        }


def keep_only_first_subword(example):
    """
    function to use in datasets.Dataset.map() for considering only the first subword
    of each word in a realignment task. This should be used simultaneously with use_first_subword_only=True
    in LabeAlignmentMapper for a token classification task
    """
    for key in ["alignment_left_positions", "alignment_right_positions"]:
        for i, (start, _) in enumerate(example[key]):
            example[key][i][1] = start + 1
    return example


class RealignmentAndOtherCollator(RealignmentCollator):
    """
    deprecated: useless with the new training loop
    Collator for building batch that contain simultaneously samples from a realignment
    task and samples for another task, handled by self.other_collator
    """

    def __init__(self, tokenizer, other_collator, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.other_collator = other_collator
        self.count_alignment = 0
        self.count_task = 0
        self.history = []

    def __call__(self, examples):
        alignment_examples = list(filter(lambda x: x.get("left_input_ids") is not None, examples))
        task_examples = list(filter(lambda x: x.get("input_ids") is not None, examples))

        self.count_alignment += len(alignment_examples)
        self.count_task += len(task_examples)

        if len(alignment_examples) > 0 and len(task_examples) > 0:
            state = "mixed"
        elif len(alignment_examples) > 0:
            state = "alignment"
        elif len(task_examples) > 0:
            state = "task"
        else:
            state = "empty"

        if len(self.history) == 0 or self.history[-1][0] != state:
            self.history.append((state, 1))
        else:
            self.history[-1] = (state, self.history[-1][1] + 1)

        if len(alignment_examples) > 0:
            try:
                alignment_batch = super(RealignmentAndOtherCollator, self).__call__(
                    alignment_examples
                )
            except Exception as e:
                raise e
        else:
            alignment_batch = {}

        if len(task_examples) > 0:
            other_inputs = [
                {
                    k: v
                    for k, v in ex.items()
                    if not k.startswith("left_")
                    and not k.startswith("right_")
                    and not k.startswith("alignment_")
                }
                for ex in task_examples
            ]
            batch_others = self.other_collator(other_inputs)
        else:
            batch_others = {}
        return {**alignment_batch, **batch_others}


def get_realignment_dataset(
    tokenizer,
    translation_dataset,
    left_lang,
    right_lang,
    dico_path=None,
    dico=None,
    mapper_for_realignment=None,
    dico_fraction=1.0,
    return_dico=False,
    first_subword_only=False,
    left_lang_id=0,
    right_lang_id=0,
    seed=None,
    max_length=None,
    ignore_identical=True,
    add_identical=False,
    split="all",
):
    """
    Build a realignment dataset from a translation dataset

    Arguments:
    - tokenizer
    - translation_dataset
    - left_lang: id of the left lang (probably 'en')
    - right_lang
    - dico_path: path to the directory containing files for dictionaires (like "en-fr.txt")
    - mapper_for_realignment: None by default, can be useful if we want to define a mapper beforehand
        and filter the dictionary (train/test split) it uses for building the alignment table
    - dico_fraction: 1. by default, smaller if we want to sample the dictionary and use return_dico=True to return the test dictionary
    - return_dico: False by default, whether to return the remaining subset of the dictionary which was not used for training realignment (if dico_fraction < 1.)
    - first_subword_only: False by default, whether to realign the representation of the first subword or an average of all subwords
    - left_lang_id: an arbitrary id for the left lang (usefull only if we use orthogonal mapping in realignment)
    - right_lang_id: same for right
    - seed
    - max_length
    - ignore_identical: whether to ignore identical words in the realignment task (default to True)
    """
    mapper = mapper_for_realignment or DatasetMapperForRealignment(
        tokenizer,
        left_lang,
        right_lang,
        dico_path=dico_path,
        dico=dico,
        dictionary_fraction=dico_fraction,
        max_length=max_length,
        ignore_identical=ignore_identical,
        add_identical=add_identical,
        split=split,
    )

    if not isinstance(translation_dataset, IterableDataset):
        translation_dataset = convert_dataset_to_iterable_dataset(translation_dataset)

    translation_dataset = translation_dataset.map(mapper, remove_columns=[left_lang, right_lang])

    translation_dataset = translation_dataset.map(
        lambda x: {**x, "left_lang_id": [left_lang_id], "right_lang_id": [right_lang_id]}
    )

    if first_subword_only:
        translation_dataset = translation_dataset.map(keep_only_first_subword)

    translation_dataset = (
        translation_dataset.filter(lambda x: len(x["alignment_left_ids"]) > 0)
        .shuffle(seed=seed)
        .with_format("torch")
    )
    if return_dico:
        return translation_dataset, mapper.other_dico
    return translation_dataset


def get_multilingual_news_commentary_realignment_dataset(
    tokenizer,
    lang_pairs: List[Tuple[str, str]],
    probabilities=None,
    dico_path=None,
    first_subword_only=True,
    lang_to_id=None,
    dataset_name="news_commentary",
    seed=None,
    cache_dir=None,
    max_length=None,
    ignore_identical=True,
    add_identical=False,
    split="all",
):
    """
    Retrieve one or several translation datasets and transform them to create a single realignment dataset

    Arguments:
    - tokenizer
    - lang_pairs: List[Tuple[str, str]], contains tuple of languages (alpha 2 code) for getting translations datasets and dictionaries
        e.g. [("en", "fr")]
    - probabilities: probabilities associated with each pair of language for when interleaving the datasets
    - dico_path: path to the directory containing files for dictionaires (like "en-fr.txt")
    - first_subword_only: True by default, whether to realign the representation of the first subword or an average of all subwords
    - lang_to_id: None by default, dictionary which attribute an id to each language, will build one if not provided, usefull only
        if we learn an orthogonal mapping during realignment
    - dataset_name: 'news_commentary' by default, 'opus100' is also supported, desings the name of the translation dataset to use
    - seed
    - cache_dir: the datasets_cache_dir for the HF load_dataset function
    - max_length
    - ignore_identical: whether to ignore identical words in the realignment task (default to True)
    """

    if dataset_name == "news_commentary":
        dataset_getter = get_news_commentary
    elif dataset_name == "opus100":
        dataset_getter = get_opus100
    else:
        raise NotImplementedError(f"dataset_name `{dataset_name}` is not expected.")
    # by convention, we fix the pivot language as first left_lang (usually English)
    pivot = lang_pairs[0][0]
    lang_to_id = lang_to_id or {
        pivot: -1,
        **{
            lang: i
            for i, lang in enumerate(
                filter(
                    lambda x: x != pivot,
                    set(
                        list(map(lambda x: x[0], lang_pairs))
                        + list(map(lambda x: x[1], lang_pairs))
                    ),
                )
            )
        },
    }
    datasets = [
        get_realignment_dataset(
            tokenizer,
            dataset_getter(left_lang, right_lang, cache_dir=cache_dir),
            left_lang,
            right_lang,
            dico_path=dico_path,
            first_subword_only=first_subword_only,
            left_lang_id=lang_to_id[left_lang],
            right_lang_id=lang_to_id[right_lang],
            seed=seed,
            max_length=max_length,
            ignore_identical=ignore_identical,
            add_identical=add_identical,
            split=split,
        )
        for i, (left_lang, right_lang) in enumerate(lang_pairs)
    ]

    return interleave_datasets(
        list(map(infinite_iterable_dataset, datasets)), probabilities=probabilities
    )


def mix_realignment_with_dataset(
    model,
    realignment_dataset,
    task_dataset,
    n_epochs=1,
    epoch_len=None,
    label_names=None,
    strategy="during",
    seed=None,
):
    """
    deprecated: useless with the new training loop
    """
    if not isinstance(task_dataset, IterableDataset):
        epoch_len = len(task_dataset)
        iterable_task_dataset = convert_dataset_to_iterable_dataset(task_dataset, repeat=n_epochs)
    else:
        if epoch_len is not None:
            task_dataset = task_dataset.take(epoch_len)
        if n_epochs > 1:
            task_dataset = repeat_iterable_dataset(task_dataset, n_epochs)
        iterable_task_dataset = task_dataset

    features = set(next(iter(iterable_task_dataset)).keys())
    expected = set(get_signature_columns_if_needed(model, label_names))

    to_remove = list(features - expected)
    if len(to_remove) > 0:
        logging.warning(
            f"Will remove columns {to_remove} from training dataset, as they are not used as input of the model"
        )
        iterable_task_dataset = iterable_task_dataset.remove_columns(to_remove)

    if strategy == "during":
        inject_mapper = DatasetMapperForInjectingRealignmentData(realignment_dataset)
        training_dataset = iterable_task_dataset.map(inject_mapper)
    # TODO add option with interleave (in distinct batches?...)
    elif strategy == "before":
        training_dataset = IterableDataset(
            enumerate(
                itertools.chain(
                    realignment_dataset.take(n_epochs * epoch_len), iterable_task_dataset
                )
            )
        )
    elif strategy == "after":
        training_dataset = IterableDataset(
            enumerate(
                itertools.chain(
                    iterable_task_dataset, realignment_dataset.take(n_epochs * epoch_len)
                )
            )
        )
    else:
        raise NotImplementedError(f"Realignment strategy not implemented: {strategy}")

    return TorchCompatibleIterableDataset(training_dataset)
