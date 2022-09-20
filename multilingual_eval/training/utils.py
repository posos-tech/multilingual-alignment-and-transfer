import logging
from torch.utils.data import DataLoader
import torch


def get_next_or_restart(dataloader: DataLoader, iterator, name=None):
    name = name or str(dataloader)
    try:
        batch = next(iterator)
    except StopIteration:
        logging.warning(f"Reached end of Dataloader {name}. Starting over")
        iterator = iter(dataloader)
        batch = next(iterator)
    return iterator, batch


def bring_batch_to_model(batch, model):
    return {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def prefix_dictionary(dictionary, prefix=None):
    if prefix is None:
        return dictionary
    return {f"{prefix}_{k}": v for k, v in dictionary.items()}
