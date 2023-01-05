from collections import defaultdict
from typing import Dict
from multilingual_eval.training.utils import bring_batch_to_model, prefix_dictionary

from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification
from transformers.trainer_pt_utils import nested_concat

from multilingual_eval.datasets.token_classification import get_token_classification_metrics


def evaluate_any_task(
    model,
    eval_dataloader,
    metric_fn,
    prefix="eval",
    remove_from_input=None,
    keep_in_input=None,
    keep_in_output=None,
    padding_index=-100,
):
    remove_from_input = remove_from_input or []
    keep_in_input = keep_in_input or ["labels"]
    keep_in_output = keep_in_output or ["logits"]

    model.eval()

    all_results = {key: None for key in keep_in_input + keep_in_output}
    for i, batch in enumerate(eval_dataloader):
        results = {key: batch[key].numpy() for key in keep_in_input}
        for key in remove_from_input:
            del batch[key]
        batch = bring_batch_to_model(batch, model)

        outputs = model(**batch, return_dict=True)

        results.update({key: outputs[key].detach().cpu().numpy() for key in keep_in_output})

        all_results = {
            key: results[key]
            if val is None
            else nested_concat(val, results[key], padding_index=padding_index)
            for key, val in all_results.items()
        }

    return prefix_dictionary(metric_fn(all_results), prefix)


def evaluate_token_classification(model, eval_dataloader, prefix="eval", metric_fn=None):
    """
    Evaluates a model on a given dataloader
    """
    model.eval()

    metric_fn = metric_fn or get_token_classification_metrics()

    all_labels = None
    all_predictions = None
    for i, batch in enumerate(eval_dataloader):
        labels = batch["labels"].numpy()
        batch = bring_batch_to_model(batch, model)

        predictions = model(**batch, return_dict=True).logits
        predictions = predictions.detach().cpu().numpy()

        all_labels = (
            labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
        )
        all_predictions = (
            predictions
            if all_predictions is None
            else nested_concat(all_predictions, predictions, padding_index=-100)
        )

    return prefix_dictionary(metric_fn((all_predictions, all_labels)), prefix)


def evaluate_several_token_classification(
    tokenizer,
    model,
    datasets,
    batch_size,
    prefixes=None,
    overall_prefix=None,
    metric_fn=None,
    collator=None,
):
    """
    Evaluates a model on several datasets, also aggregates the metrics with prefix "avg".
    Metrics will have prefixes of the form {overall_prefix}_{prefixes[i] or avg}_name_of_the_metrics
    """
    collator = collator or DataCollatorForTokenClassification(tokenizer)
    prefixes = prefixes or [str(i) for i in range(len(datasets))]
    assert len(datasets) == len(prefixes)
    assert "avg" not in prefixes

    dataloaders = [
        DataLoader(dataset, batch_size=batch_size, collate_fn=collator) for dataset in datasets
    ]
    res = {}
    agg = defaultdict(lambda: 0)
    for dataloader, prefix in zip(dataloaders, prefixes):
        next_res = evaluate_token_classification(
            model, dataloader, prefix=None, metric_fn=metric_fn
        )
        for key, value in next_res.items():
            agg[key] += value
        res.update(prefix_dictionary(next_res, prefix))

    res.update(prefix_dictionary({k: v / len(datasets) for k, v in agg.items()}, prefix="avg"))

    return prefix_dictionary(res, overall_prefix)
