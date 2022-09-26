from typing import List, Union
import numpy as np

from datasets import load_dataset, interleave_datasets, load_metric
from multilingual_eval.datasets.code_switching import (
    get_dataset_with_code_swicthing,
)
from multilingual_eval.datasets.data_utils import convert_dataset_to_iterable_dataset

from multilingual_eval.datasets.label_alignment import LabelAlignmentMapper


def get_wikiann_ner(
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
    max_length=512,
):

    if not isinstance(lang, list):
        lang = [lang]
    if lang_id is not None:
        if not isinstance(lang_id, list):
            lang_id = [lang_id]
        assert len(lang_id) == len(lang)

    if dictionaries_for_code_switching and not isinstance(dictionaries_for_code_switching[0], list):
        dictionaries_for_code_switching = [dictionaries_for_code_switching]

    datasets = [
        load_dataset(
            "wikiann",
            elt,
            cache_dir=datasets_cache_dir,
        )[split]
        for elt in lang
    ]

    n_datasets = len(datasets)

    if limit:
        limits = [
            limit // n_datasets + (1 if i < limit % n_datasets else 0) for i in range(n_datasets)
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
            lambda x: get_dataset_with_code_swicthing(x[1], dictionaries_for_code_switching[x[0]]),
            enumerate(datasets),
        )

    datasets = list(
        map(
            lambda x: x.map(
                LabelAlignmentMapper(
                    tokenizer,
                    label_name="ner_tags",
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
                lambda x: x.remove_columns(["tokens", "ner_tags", "langs", "spans"]),
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


def get_wikiann_metric_fn():
    metric = load_metric("seqeval")

    str_labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [str_labels[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [str_labels[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    return compute_metrics
