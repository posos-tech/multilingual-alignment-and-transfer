from datasets.iterable_dataset import IterableDataset
import torch
import itertools
import inspect


def convert_dataset_to_iterable_dataset(dataset, repeat=1):
    return IterableDataset(
        enumerate(
            itertools.chain(*[dataset if i == 0 else dataset.shuffle() for i in range(repeat)])
        )
    )


def repeat_iterable_dataset(dataset, repeat):
    return convert_dataset_to_iterable_dataset(dataset, repeat)


def infinite_iterable_dataset(dataset):
    return IterableDataset(enumerate(itertools.cycle(dataset)))


def get_signature_columns_if_needed(model, label_names=None):
    # taken from HF trainer:
    # https://github.com/huggingface/transformers/blob/f0d496828d3da3bf1e3c8fbed394d7847e839fa6/src/transformers/trainer.py#L705
    label_names = label_names or []
    signature = inspect.signature(model.forward)
    signature_columns = list(signature.parameters.keys())
    signature_columns += ["label", "label_ids", "labels", *label_names]
    return signature_columns


class TorchCompatibleIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, normal_dataset):
        self.normal_dataset = normal_dataset

    def __iter__(self):
        return iter(self.normal_dataset)
