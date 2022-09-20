import logging
from torch.utils.data import DataLoader, DistributedSampler
import torch
from transformers import DataCollatorForTokenClassification


from multilingual_eval.training.epoch_loop import during_strategy_epoch_loop
from multilingual_eval.datasets.realignment_dataset import (
    RealignmentAndOtherCollator,
)
from multilingual_eval.training.evaluation_loops import evaluate_several_token_classification


def during_strategy_training_loop(
    tokenizer,
    model,
    task_dataset: DataLoader,
    realignment_dataset: DataLoader,
    evaluation_datasets=None,
    evaluation_prefixes=None,
    task_batch_size=4,
    realignment_batch_size=2,
    n_epochs=10,
    accumulation_steps=1,
    logging_steps=100,
    log_in_wandb=False,
):
    if log_in_wandb:
        import wandb

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    task_dataloader = DataLoader(
        task_dataset,
        shuffle=True,
        batch_size=task_batch_size,
        collate_fn=DataCollatorForTokenClassification(tokenizer),
    )
    realignment_dataloader = DataLoader(
        realignment_dataset,
        shuffle=False,
        batch_size=realignment_batch_size,
        collate_fn=RealignmentAndOtherCollator(
            tokenizer,
            DataCollatorForTokenClassification(tokenizer),
        ),
    )

    for i in range(n_epochs):

        during_strategy_epoch_loop(
            model,
            optimizer,
            task_dataloader,
            realignment_dataloader,
            task_accumulation_steps=accumulation_steps,
            logging_steps=logging_steps,
            log_in_wandb=log_in_wandb,
        )

        if evaluation_datasets is not None:
            res = evaluate_several_token_classification(
                tokenizer,
                model,
                evaluation_datasets,
                batch_size=task_batch_size,
                prefixes=evaluation_prefixes,
                overall_prefix="eval",
            )
            logging.info(res)
            if log_in_wandb:
                wandb.log(res)


def before_strategy_training_loop(
    model, task_dataloader: DataLoader, realignment_dataloader: DataLoader, n_epochs=10
):
    pass


def after_strategy_training_loop(
    model, task_dataloader: DataLoader, realignment_dataloader: DataLoader, n_epochs=10
):
    pass


def realignment_only_training_loop(model, realignment_dataloader: DataLoader, n_epochs=10):
    pass


def task_only_training_loop(model, task_dataloader: DataLoader, n_epochs=10):
    pass
