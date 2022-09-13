from typing import Dict, List, Set, Union
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
import numpy as np

from multilingual_eval.datasets.data_utils import (
    TorchCompatibleIterableDataset,
    convert_dataset_to_iterable_dataset,
)


class CodeSwitchingMapper:
    def __init__(
        self,
        dictionaries: List[Dict[str, Set[str]]],
        probs=None,
        replace_prob=0.5,
    ):
        self._dictionaries = dictionaries
        self._dico_probs = probs or [1.0 / len(self._dictionaries)] * len(self._dictionaries)
        self.replace_prob = replace_prob

    def process_token(self, token):
        dictionaries = []
        probs = []

        for dico, p in zip(self._dictionaries, self._dico_probs):
            if token in dico:
                dictionaries.append(dico)
                probs.append(p)

        if len(dictionaries) == 0:
            return token

        probs = list(map(lambda x: x / sum(probs), probs))

        chosen_idx = np.random.choice(len(dictionaries), p=probs)

        chosen_dictionary = dictionaries[chosen_idx]

        if self.replace_prob >= np.random.random():
            candidates = list(chosen_dictionary[token])
            replacement_idx = np.random.choice(len(candidates))
            return candidates[replacement_idx]
        return token

    def __call__(self, sample):
        new_tokens = []
        for token in sample["tokens"]:
            new_tokens.append(self.process_token(token))

        sample["tokens"] = new_tokens
        return sample


def get_dataset_with_code_swicthing(dataset, dictionaries, probs=None, replace_prob=0.5):
    mapper = CodeSwitchingMapper(dictionaries, probs=probs, replace_prob=replace_prob)

    if isinstance(dataset, (HFDataset, Dataset)):
        dataset = convert_dataset_to_iterable_dataset(dataset)

    return dataset.map(mapper)
