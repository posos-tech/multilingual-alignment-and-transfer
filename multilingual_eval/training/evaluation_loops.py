from collections import defaultdict
from multilingual_eval.training.utils import bring_batch_to_model, prefix_dictionary

from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification

from multilingual_eval.utils import get_metric_fn


def evaluate_token_classification(model, eval_dataloader, prefix="eval"):
    model.eval()

    metric_fn = get_metric_fn()

    agg = defaultdict(lambda: 0)
    divisor = defaultdict(lambda: 0)
    for i, batch in enumerate(eval_dataloader):
        labels = batch["labels"].numpy()
        batch = bring_batch_to_model(batch, model)

        predictions = model(**batch, return_dict=True).logits
        predictions = predictions.detach().cpu().numpy()

        next_res = metric_fn((predictions, labels))
        for key, value in next_res.items():
            agg[key] += value * predictions.shape[0]
            divisor[key] += predictions.shape[0]

    return prefix_dictionary({k: v / divisor[k] for k, v in agg.items()}, prefix)


def evaluate_several_token_classification(
    tokenizer, model, datasets, batch_size, prefixes=None, overall_prefix=None
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
        next_res = evaluate_token_classification(model, dataloader, prefix=None)
        for key, value in next_res.items():
            agg[key] += value
        res.update(prefix_dictionary(next_res, prefix))

    res.update(prefix_dictionary({k: v / len(datasets) for k, v in agg.items()}, prefix="avg"))

    return prefix_dictionary(res, overall_prefix)
