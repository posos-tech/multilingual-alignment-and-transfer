from transformers import XLMRobertaTokenizer, XLMRobertaTokenizerFast
from datasets.iterable_dataset import IterableDataset, ExamplesIterable
from datasets import interleave_datasets
from typing import List, Tuple, Optional, Dict
import os
from collections import defaultdict

from multilingual_eval.datasets.data_utils import TorchCompatibleIterableDataset


def get_pharaoh_dataset(
    translation_file: str,
    alignment_file: str,
    repeat=False,
    keep_only_one_to_one=True,
    ignore_identical=True,
):
    def pharaoh_reader():
        with open(translation_file) as translation_reader, open(alignment_file) as alignment_reader:
            for i, (translation, alignment) in enumerate(zip(translation_reader, alignment_reader)):
                parts = translation.split("|||")
                if len(parts) != 2:
                    continue
                left_sent, right_sent = parts
                left_tokens = left_sent.strip().split()
                right_tokens = right_sent.strip().split()

                pairs = alignment.strip().split()
                pairs = list(map(lambda x: tuple(map(int, x.split("-"))), pairs))

                left_positions = list(map(lambda x: x[0], pairs))
                right_positions = list(map(lambda x: x[1], pairs))

                if keep_only_one_to_one:
                    new_left_positions = []
                    new_right_positions = []
                    for a, b in zip(left_positions, right_positions):
                        if left_positions.count(a) > 1 or right_positions.count(b) > 1:
                            continue
                        new_left_positions.append(a)
                        new_right_positions.append(b)
                    left_positions = new_left_positions
                    right_positions = new_right_positions

                if ignore_identical:
                    new_left_positions = []
                    new_right_positions = []
                    for a, b in zip(left_positions, right_positions):
                        if left_tokens[a] == right_tokens[b]:
                            continue
                        new_left_positions.append(a)
                        new_right_positions.append(b)
                    left_positions = new_left_positions
                    right_positions = new_right_positions

                if len(left_positions) == 0:
                    continue

                yield i, {
                    "left_tokens": left_tokens,
                    "right_tokens": right_tokens,
                    "aligned_left_ids": left_positions,
                    "aligned_right_ids": right_positions,
                }

        if repeat:
            yield from pharaoh_reader()

    return IterableDataset(ExamplesIterable(pharaoh_reader, {}))


class AdaptAlignmentToTokenizerMapper:
    """
    Class for a dataset mapper that adapts word positions from the DatasetMapperForRealignment
    according to a model-specific tokenization, a bit like when realigning labels of a token-wise
    classification
    """

    def __init__(
        self, tokenizer, max_length=None, remove_underscore_if_roberta=True, first_subword_only=True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.first_subword_only = first_subword_only
        self.remove_underscore = remove_underscore_if_roberta and isinstance(
            self.tokenizer, (XLMRobertaTokenizer, XLMRobertaTokenizerFast)
        )
        if self.remove_underscore:
            self.underscore_id = self.tokenizer.convert_tokens_to_ids(["▁"])[0]

    def __call__(self, examples):
        """
        Take batch of sentences with aligned words extracted from a translation dataset
        (left_tokens, right_tokens, aligned_left_ids, and aligned_right_ids) and return
        a sample for a realignment task, with properties:
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
        left_tokens = examples["left_tokens"]
        right_tokens = examples["right_tokens"]
        aligned_left_ids = examples["aligned_left_ids"]
        aligned_right_ids = examples["aligned_right_ids"]

        left_tokenized_inputs = self.tokenizer(
            left_tokens, truncation=True, is_split_into_words=True, max_length=self.max_length
        )
        right_tokenized_inputs = self.tokenizer(
            right_tokens, truncation=True, is_split_into_words=True, max_length=self.max_length
        )

        alignment_left_pos, new_left_inputs = self.retrieve_positions_and_remove_underscore(
            left_tokenized_inputs, left_tokens
        )
        alignment_right_pos, new_right_inputs = self.retrieve_positions_and_remove_underscore(
            right_tokenized_inputs, right_tokens
        )

        (
            alignment_left_pos,
            alignment_right_pos,
            aligned_left_ids,
            aligned_right_ids,
        ) = self.realign_pairs(
            alignment_left_pos, alignment_right_pos, aligned_left_ids, aligned_right_ids
        )

        return {
            **{f"left_{k}": v for k, v in new_left_inputs.items()},
            **{f"right_{k}": v for k, v in new_right_inputs.items()},
            "alignment_left_ids": aligned_left_ids,
            "alignment_right_ids": aligned_right_ids,
            "alignment_left_positions": alignment_left_pos,
            "alignment_right_positions": alignment_right_pos,
            "alignment_nb": [len(elt) for elt in aligned_left_ids],
            "alignment_left_length": [len(elt) for elt in alignment_left_pos],
            "alignment_right_length": [len(elt) for elt in alignment_right_pos],
        }

    def retrieve_positions_and_remove_underscore(self, tokenized_inputs, tokens):
        """
        Retrieve positions of words in term of subword and remove special Roberta underscore
        if needed
        """
        if self.remove_underscore:
            result = {key: [] for key in tokenized_inputs}
        else:
            result = tokenized_inputs

        n = len(tokenized_inputs["input_ids"])

        positions = [[None for _ in sent] for sent in tokens]
        for i in range(n):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            if self.remove_underscore:
                result_sample = {key: [] for key in tokenized_inputs if key in result}
            delta = 0
            for j, word_idx in enumerate(word_ids):

                if self.remove_underscore:
                    if tokenized_inputs["input_ids"][i][j] == self.underscore_id:
                        delta += -1
                        continue
                    for key in result_sample:
                        result_sample[key].append(tokenized_inputs[key][i][j])

                if word_idx is None:
                    continue

                if positions[i][word_idx] is None:
                    positions[i][word_idx] = [j + delta, j + delta + 1]
                elif not self.first_subword_only:
                    positions[i][word_idx][1] = j + delta + 1

            if self.remove_underscore:
                for key, value in result_sample.items():
                    result[key].append(value)

        return positions, result

    def realign_pairs(
        self, alignment_left_pos, alignment_right_pos, alignment_left_ids, alignment_right_ids
    ):
        """
        Remove null values in positions of words (in alignment_left_pos and alignment_right_pos)
        and modifies alignment_left_ids and alignment_right_ids accordingly
        """

        new_alignment_left_pos = []
        new_alignment_right_pos = []
        new_alignment_left_ids = []
        new_alignment_right_ids = []

        n = len(alignment_left_pos)
        for i in range(n):
            alignment_left_pos_entry = []
            alignment_right_pos_entry = []

            left_old_to_new_pos = []
            right_old_to_new_pos = []

            for pos in alignment_left_pos[i]:
                if pos is None:
                    left_old_to_new_pos.append(None)
                else:
                    alignment_left_pos_entry.append(pos)
                    left_old_to_new_pos.append(len(alignment_left_pos_entry) - 1)

            for pos in alignment_right_pos[i]:
                if pos is None:
                    right_old_to_new_pos.append(None)
                else:
                    alignment_right_pos_entry.append(pos)
                    right_old_to_new_pos.append(len(alignment_right_pos_entry) - 1)

            alignment_left_ids_entry = []
            alignment_right_ids_entry = []

            for left_idx, right_idx in zip(alignment_left_ids[i], alignment_right_ids[i]):
                if (
                    left_old_to_new_pos[left_idx] is not None
                    and right_old_to_new_pos[right_idx] is not None
                ):
                    alignment_left_ids_entry.append(left_old_to_new_pos[left_idx])
                    alignment_right_ids_entry.append(right_old_to_new_pos[right_idx])

            new_alignment_left_pos.append(alignment_left_pos_entry)
            new_alignment_right_pos.append(alignment_right_pos_entry)
            new_alignment_left_ids.append(alignment_left_ids_entry)
            new_alignment_right_ids.append(alignment_right_ids_entry)

        return (
            new_alignment_left_pos,
            new_alignment_right_pos,
            new_alignment_left_ids,
            new_alignment_right_ids,
        )


def get_realignment_dataset_for_one_pair(
    tokenizer,
    translation_file: str,
    alignment_file: str,
    max_length=None,
    first_subowrd_only=True,
    ignore_identical=True,
    seed=None,
    left_id=None,
    right_id=None,
):
    """
    Get realignment dataset from a parallel dataset in FastAlign format and FastAlign output
    """
    raw_dataset = get_pharaoh_dataset(
        translation_file, alignment_file, ignore_identical=ignore_identical
    ).shuffle(seed=seed, buffer_size=10_000)
    mapper = AdaptAlignmentToTokenizerMapper(
        tokenizer, max_length=max_length, first_subword_only=first_subowrd_only
    )
    dataset = raw_dataset.map(
        mapper,
        batched=True,
        remove_columns=["aligned_left_ids", "aligned_right_ids", "left_tokens", "right_tokens"],
    ).filter(lambda x: len(x["alignment_left_ids"]) > 0)
    if left_id is not None:
        dataset = dataset.map(lambda x: {**x, "left_lang_id": [left_id]})
    if right_id is not None:
        dataset = dataset.map(lambda x: {**x, "right_lang_id": [right_id]})
    return dataset.with_format("torch")


def get_multilingual_realignment_dataset(
    tokenizer,
    translation_path: str,
    alignment_path: str,
    pairs: List[Tuple[str, str]],
    max_length=None,
    seed=None,
    lang_to_id: Optional[Dict[str, int]] = None,
    split="train",
    return_torch_compatible=True,
):
    lang_to_id = lang_to_id or defaultdict(lambda: None)
    datasets = [
        get_realignment_dataset_for_one_pair(
            tokenizer,
            os.path.join(translation_path, f"{left_lang}-{right_lang}.tokenized.{split}.txt"),
            os.path.join(alignment_path, f"{left_lang}-{right_lang}.{split}"),
            max_length=max_length,
            seed=seed,
            left_id=lang_to_id[left_lang],
            right_id=lang_to_id[right_lang],
        )
        for left_lang, right_lang in pairs
    ]

    dataset = interleave_datasets(datasets)

    if return_torch_compatible:
        dataset = TorchCompatibleIterableDataset(dataset)

    return dataset
