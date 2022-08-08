# Chinese load_dataset("un_multi", "en-zh", cache_dir=cache_dir)
# Others news_commentary
from collections import defaultdict
from transformers import DataCollatorWithPadding
from dataclasses import dataclass
from typing import Dict, Set, Tuple
from time import perf_counter
import torch
from torch.utils.data import DataLoader

from multilingual_eval.data import get_dicos
from multilingual_eval.utils import get_tokenizer_type, subwordlist_to_wordlist


@dataclass
class BilingualDictionary:
    forward: Dict[Tuple[str], Set[Tuple[str]]]
    backward: Dict[Tuple[str], Set[Tuple[str]]]


def timing(func):
    def wrapper(self, *args, **kwargs):
        start = perf_counter()
        res = func(self, *args, **kwargs)
        end = perf_counter()
        self.perfs[func.__name__] = [
            self.perfs[func.__name__][0] + end - start,
            self.perfs[func.__name__][1] + 1,
        ]
        return res

    return wrapper


class DatasetMapperForRealignment:
    def __init__(self, tokenizer, dico_path: str, left_key: str, right_key: str):
        self.tokenizer = tokenizer

        forward, backward = get_dicos(
            left_key, right_key, dico_path, ignore_identical=True, tokenizer=tokenizer
        )

        self.dico = BilingualDictionary(forward, backward)

        self.left_key = left_key
        self.right_key = right_key

        self._max_left_length = max(map(len, forward))
        self._max_right_length = max(map(len, backward))

        self._tokenizer_type = get_tokenizer_type(tokenizer)

        self.perfs = defaultdict(lambda: [0.0, 0])

    @timing
    def tokenize_sentences(self, left_sent, right_sent):

        if left_sent is None or right_sent is None:
            raise Exception(
                "null sentence found in dataset. Please filter translation datasets for null sentences"
            )

        left_tokens = self.tokenizer.tokenize(left_sent)
        right_tokens = self.tokenizer.tokenize(right_sent)

        return left_tokens, right_tokens

    @timing
    def create_multi_subwords(self, left_tokens, right_tokens):
        # Generate potential multi-word expression (essentially for languages like mandarin chinese
        # which are tokenized by character and have words spanning several tokens)
        left_multi = []
        left_multi_pos = []
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

    @timing
    def find_aligned_words(self, left_multi, left_multi_pos, right_multi, right_multi_pos):
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

    @timing
    def remove_overlapping_aligned(self, aligned_left_multi_pos, aligned_right_multi_pos):
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

    @timing
    def create_words_from_subwords(self, left_tokens, right_tokens):
        _, left_word_positions = subwordlist_to_wordlist(
            left_tokens, split_type=self._tokenizer_type
        )
        _, right_word_positions = subwordlist_to_wordlist(
            right_tokens, split_type=self._tokenizer_type
        )
        return left_word_positions, right_word_positions

    @timing
    def create_final_subword_list(
        self,
        left_word_positions,
        right_word_positions,
        aligned_left_multi_pos,
        aligned_right_multi_pos,
    ):
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

        for start, end in left_word_positions:
            if start not in left_covered_ids and end - 1 not in left_covered_ids:
                alignment_left_positions.append([start, end])

        alignment_left_ids = list(
            range(
                len(alignment_left_positions),
                len(alignment_left_positions) + len(aligned_left_multi_pos),
            )
        )
        alignment_left_positions += aligned_left_multi_pos

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
        left_sent = example[self.left_key]
        right_sent = example[self.right_key]

        left_tokens, right_tokens = self.tokenize_sentences(left_sent, right_sent)

        left_multi, left_multi_pos, right_multi, right_multi_pos = self.create_multi_subwords(
            left_tokens, right_tokens
        )

        aligned_left_multi_pos, aligned_right_multi_pos = self.find_aligned_words(
            left_multi, left_multi_pos, right_multi, right_multi_pos
        )

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
            **{f"left_{k}": v for k, v in self.tokenizer(left_sent, truncation=True).items()},
            **{f"right_{k}": v for k, v in self.tokenizer(right_sent, truncation=True).items()},
            "alignment_left_ids": alignment_left_ids,
            "alignment_right_ids": alignment_right_ids,
            "alignment_left_positions": alignment_left_positions,
            "alignment_right_positions": alignment_right_positions,
            "alignment_nb": len(alignment_left_ids),
            "alignment_left_length": len(alignment_left_positions),
            "alignment_right_length": len(alignment_right_positions),
        }


class RealignmentCollator:
    def __init__(self, tokenizer):
        self.usual_collator = DataCollatorWithPadding(tokenizer)

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
        alignment_left_positions = torch.zeros((len(examples), max_left_length, 2), dtype=torch.long)
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


def get_realignment_dataset(tokenizer, translation_dataset, left_lang, right_lang, dico_path):
    mapper = DatasetMapperForRealignment(tokenizer, dico_path, left_lang, right_lang)

    translation_dataset = (
        translation_dataset.map(mapper, remove_columns=[left_lang, right_lang])
        .filter(lambda x: len(x["alignment_left_ids"]) > 0)
        .shuffle()
        .with_format("torch")
    )
    return translation_dataset


def get_realignment_dataloader(
    tokenizer, translation_dataset, left_lang, right_lang, dico_path, batch_size: int
):
    translation_dataset = get_realignment_dataset(
        tokenizer, translation_dataset, left_lang, right_lang, dico_path
    )

    return DataLoader(
        translation_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=RealignmentCollator(tokenizer),
    )
