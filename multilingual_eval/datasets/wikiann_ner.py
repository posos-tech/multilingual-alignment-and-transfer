import logging
from typing import List, Union
import numpy as np

from datasets import load_dataset, load_metric


from multilingual_eval.datasets.token_classification import get_token_classification_getter


get_wikiann_ner = get_token_classification_getter(
    lambda lang, cache_dir=None: load_dataset(
        "wikiann",
        lang,
        cache_dir=cache_dir,
    ),
    "ner_tags",
)


def ner_post_processing(labels):

    new_labels = []

    # Replace standalone I-X by B-X
    previous = None
    for i, label in enumerate(labels):
        if label[0] == "I" and (previous is None or previous == "O"):
            new_labels.append(f"B-{label[2:]}")
        else:
            new_labels.append(label)
        previous = label

    # Replace B-X I-Y I-Z by B-Z I-Z I-Z
    previous = None
    for i, label in zip(list(range(len(new_labels)))[::-1], new_labels[::-1]):
        if previous is None and label[0] == "I":
            previous = label[2:]
        elif label == "O":
            previous = None
        elif previous is not None and label[2:] != previous:
            new_labels[i] = f"{label[0]}-{previous}"
            if label[0] == "B":
                previous = None

    return new_labels


def get_wikiann_metric_fn():
    """
    Get dedicated metrics for the wikiann dataset
    """

    metric = load_metric("seqeval")

    str_labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        raw_true_predictions = [
            [str_labels[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        true_predictions = list(map(ner_post_processing, raw_true_predictions))

        true_labels = [
            [str_labels[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        logging.debug(f"raw predictions: {raw_true_predictions[:5]}")
        logging.debug(f"post-processed predictions: {true_predictions[:5]}")
        logging.debug(f"labels: {true_labels[:5]}")

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    return compute_metrics
