from transformers import (
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    DataCollatorForTokenClassification,
)

from multilingual_eval.datasets.wikiann_ner import get_wikiann_ner, get_wikiann_metric_fn
from multilingual_eval.datasets.xnli import get_xnli, xnli_metric_fn
from multilingual_eval.datasets.xtreme_udpos import get_wuetal_udpos
from multilingual_eval.datasets.pawsx import get_pawsx, pawsx_metric_fn
from multilingual_eval.datasets.token_classification import get_token_classification_metrics

from multilingual_eval.models.with_realignment_factory import (
    AutoModelForSequenceClassificationWithRealignment,
    AutoModelForTokenClassificationWithRealignment,
)


def get_dataset_fn(name, zh_segmenter=None):
    return {
        "wikiann": lambda *args, **kwargs: get_wikiann_ner(
            *args, **kwargs, zh_segmenter=zh_segmenter
        ),
        "udpos": get_wuetal_udpos,
        "xnli": get_xnli,
        "pawsx": get_pawsx,
    }[name]


def get_dataset_metric_fn(name):
    return {
        "wikiann": get_wikiann_metric_fn,
        "udpos": get_token_classification_metrics,
        "xnli": lambda: xnli_metric_fn,
        "pawsx": lambda: pawsx_metric_fn,
    }[name]


def get_model_class_for_dataset_with_realignment(name):
    return {
        "wikiann": AutoModelForTokenClassificationWithRealignment,
        "udpos": AutoModelForTokenClassificationWithRealignment,
        "xnli": AutoModelForSequenceClassificationWithRealignment,
        "pawsx": AutoModelForSequenceClassificationWithRealignment,
    }[name]


def model_fn(task_name, with_realignment=False):
    if with_realignment:
        token_classification = AutoModelForTokenClassificationWithRealignment
        sequence_classification = AutoModelForSequenceClassificationWithRealignment
    else:
        token_classification = AutoModelForTokenClassification
        sequence_classification = AutoModelForSequenceClassification
    return {
        "wikiann": lambda *args, **kwargs: token_classification.from_pretrained(
            *args, **kwargs, num_labels=7
        ),
        "udpos": lambda *args, **kwargs: token_classification.from_pretrained(
            *args, **kwargs, num_labels=18
        ),
        "xnli": lambda *args, **kwargs: sequence_classification.from_pretrained(
            *args, **kwargs, num_labels=3
        ),
        "pawsx": lambda *args, **kwargs: sequence_classification.from_pretrained(
            *args, **kwargs, num_labels=3
        ),
    }[task_name]


def collator_fn(task_name):
    if task_name in ["wikiann", "udpos"]:
        return DataCollatorForTokenClassification
    elif task_name in ["xnli", "pawsx"]:
        return DataCollatorWithPadding
    raise KeyError(task_name)
