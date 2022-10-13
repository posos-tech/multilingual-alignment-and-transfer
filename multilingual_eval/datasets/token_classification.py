from typing import Union, List
import numpy as np
import pycountry

from datasets import interleave_datasets
from multilingual_eval.datasets.code_switching import (
    get_dataset_with_code_swicthing,
)
from multilingual_eval.datasets.data_utils import convert_dataset_to_iterable_dataset

from multilingual_eval.datasets.label_alignment import LabelAlignmentMapper


def get_token_classification_getter(subset_loader, label_name: str, to_remove=None):
    """
    Return a function that would load a token classification dataset and perform the
    necessary transformation

    Arguments:
    - subset_loader: a function that takes two arguments 'lang' (positional) and 'cache_dir' (keyword) that will
        load the dataset for a given language (lang) using the provided cache directory (cache_dir) which is None by default
    - label_name: the name of the properties containing the labels as integers
    """
    to_remove = to_remove or []

    def get_token_classification_dataset(
        lang: Union[List[str], str],
        tokenizer,
        limit=None,
        split="train",
        datasets_cache_dir=None,
        interleave=True,
        first_subword_only=True,
        lang_id=None,
        dictionaries_for_code_switching=None,
        return_length=False,
        n_epochs=1,
        remove_useless=True,
        max_length=None,
    ):
        """
        Load a task classification dataset for a given lang

        Arguments:

        - lang: the language, can be a list of labels if we want to load several subsets
        - tokenizer
        - limit: a limit on the total number of samples, default to None (no limit)
        - split: the split (typically "train" or "validation"), default "train"
        - datasets_cache_dir: the cache directory for the load_dataset function
        - interleave: if several languages are provided, decides whether to interleave the different
            datasets or return them as element of a list (default to True)
        - first_subword_only: whether to perform classification on the first subword of each token
            or on each subword (default to True)
        """

        if not isinstance(lang, list):
            lang = [lang]
        if lang_id is not None:
            if not isinstance(lang_id, list):
                lang_id = [lang_id]
            assert len(lang_id) == len(lang)

        if dictionaries_for_code_switching and not isinstance(
            dictionaries_for_code_switching[0], list
        ):
            dictionaries_for_code_switching = [dictionaries_for_code_switching]

        datasets = [subset_loader(elt, cache_dir=datasets_cache_dir)[split] for elt in lang]

        n_datasets = len(datasets)

        if limit:
            limits = [
                limit // n_datasets + (1 if i < limit % n_datasets else 0)
                for i in range(n_datasets)
            ]

            datasets = map(
                lambda x: x[0].shuffle().filter(lambda _, i: i < x[1], with_indices=True),
                zip(datasets, limits),
            )

        if n_datasets == 1:
            datasets = [next(iter(datasets))]
        elif interleave:
            datasets = [interleave_datasets(datasets)]

        if return_length:
            lengths = list(map(len, datasets))

        if n_epochs > 1:
            datasets = map(lambda x: convert_dataset_to_iterable_dataset(x, n_epochs), datasets)

        if dictionaries_for_code_switching:
            datasets = map(
                lambda x: get_dataset_with_code_swicthing(
                    x[1], dictionaries_for_code_switching[x[0]]
                ),
                enumerate(datasets),
            )

        datasets = list(
            map(
                lambda x: x.map(
                    LabelAlignmentMapper(
                        tokenizer,
                        label_name=label_name,
                        first_subword_only=first_subword_only,
                        max_length=max_length,
                    ),
                    batched=True,
                ),
                datasets,
            ),
        )

        if lang_id is not None:
            datasets = list(
                map(lambda x: x[0].map(lambda y: {**y, "lang_id": [x[1]]}), zip(datasets, lang_id))
            )

        if remove_useless:
            datasets = list(
                map(
                    lambda x: x.remove_columns(["tokens", label_name] + to_remove),
                    datasets,
                )
            )

        if n_datasets == 1 or interleave:
            if return_length:
                return datasets[0], lengths[0]
            return datasets[0]
        if return_length:
            return datasets, lengths
        return datasets

    return get_token_classification_dataset


def get_token_classification_metrics():
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            p
            for prediction, label in zip(predictions, labels)
            for (p, l) in zip(prediction, label)
            if l != -100
        ]
        true_labels = [
            l
            for prediction, label in zip(predictions, labels)
            for (p, l) in zip(prediction, label)
            if l != -100
        ]

        return {
            "accuracy": np.mean(
                list(map(lambda x: x[0] == x[1], zip(true_labels, true_predictions)))
            )
        }

    return compute_metrics
