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
    to_remove=["langs", "spans"],
)


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
