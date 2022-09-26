from collections import defaultdict
from typing import Dict
from multilingual_eval.training.utils import bring_batch_to_model, prefix_dictionary

from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification, TrainingArguments
from transformers.trainer_pt_utils import nested_concat

from multilingual_eval.trainer import TrainerWithSeveralEvalDatasets
from multilingual_eval.utils import get_metric_fn


def evaluate_token_classification(model, eval_dataloader, prefix="eval", metric_fn=None):
    model.eval()

    metric_fn = metric_fn or get_metric_fn()

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
    tokenizer, model, datasets, batch_size, prefixes=None, overall_prefix=None, metric_fn=None
):
    prefixes = prefixes or [str(i) for i in range(len(datasets))]
    assert len(datasets) == len(prefixes)
    assert "avg" not in prefixes

    dataloaders = [
        DataLoader(
            dataset, batch_size=batch_size, collate_fn=DataCollatorForTokenClassification(tokenizer)
        )
        for dataset in datasets
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
