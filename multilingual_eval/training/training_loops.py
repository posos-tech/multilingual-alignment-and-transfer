import logging
from torch.utils.data import DataLoader, DistributedSampler
import torch
from transformers import DataCollatorForTokenClassification
from transformers.optimization import AdamW, get_scheduler
import numpy as np
import random

from multilingual_eval.training.epoch_loop import epoch_loop
from multilingual_eval.datasets.realignment_dataset import (
    RealignmentAndOtherCollator,
)
from multilingual_eval.training.evaluation_loops import (
    evaluate_several_token_classification,
    evaluate_token_classification,
)


def realignment_training_loop(
    tokenizer,
    model,
    task_dataset: DataLoader,
    realignment_dataset: DataLoader,
    strategy="during",
    evaluation_datasets=None,
    same_language_evaluation_dataset=None,
    evaluation_prefixes=None,
    task_batch_size=4,
    realignment_batch_size=2,
    n_epochs=10,
    accumulation_steps=1,
    logging_steps=100,
    log_in_wandb=False,
    metric_fn=None,
    realignment_coef=0.1,
    realignment_coef_scheduler=None,
    data_collator=None,
    seed=None,
):
    """
    Performs a training loop, with or without realignment

    Arguments:

    - tokenizer
    - model
    - task_dataset: training dataset for the fine-tuning task (must have a length)
    - realignment_dataset: iterable dataset for the realignment auxiliary task
    - strategy: default "during", the realignment strategy (either "baseline" for no realignment, or "after", "before" or "during")
    - evaluation_datasets: optional list of evaluation datasets
    - same_language_evaluation_dataset: optional evaluation dataset on same language as training
    - evaluation_prefixes: optional list of prefixes for evaluation datasets metrics
    - task_batch_size: batch size for the training task (not considering accumulation steps)
    - realignment_batch_size: batch size of the realignment step
    - n_epochs: number of epochs for the fine-tuning task
    - accumulation_steps: number of accumulation steps for the fine-tuning task
    - logging_steps: int, default 100, number of steps (in term of optimization steps, hence nb of batch / accumulation steps) between each log of training stats
    - log_in_wandb: whether to log training stats in wandb (conditional import)
    - metric_fn: function that gets the metric from the overall predictions and labels
    - realignment_coef: float, default 0.1, the coefficient to apply to the realignment loss
    - realignment_coef_scheduler: a function that takes an integer (the epoch) and return a float, the coefficient to apply to the realignment loss at
        given epochs, overrides realignment_coef
    - data_collator: default None, if None, will default to DataCollatorForTokenClassification(tokenizer)
    """
    data_collator = data_collator or DataCollatorForTokenClassification(tokenizer)

    if log_in_wandb:
        import wandb

    if model.device.type != "cuda" and torch.cuda.device_count() > 0:
        model = model.to(0)

    # define optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.999), eps=1e-8)

    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

    else:
        g = None
        seed_worker = None
    task_dataloader = DataLoader(
        task_dataset,
        shuffle=True,
        batch_size=task_batch_size,
        collate_fn=data_collator,
        worker_init_fn=seed_worker,
        generator=g,
    )
    if strategy != "baseline":
        realignment_dataloader = DataLoader(
            realignment_dataset,
            shuffle=False,
            batch_size=realignment_batch_size,
            collate_fn=RealignmentAndOtherCollator(
                tokenizer,
                data_collator,
            ),
        )
    else:
        realignment_dataloader = None

    if same_language_evaluation_dataset is not None:
        same_language_evaluation_dataloader = DataLoader(
            same_language_evaluation_dataset,
            shuffle=False,
            batch_size=task_batch_size,
            collate_fn=data_collator,
        )

    if strategy == "before":
        for i in range(n_epochs):
            epoch_loop(
                model,
                optimizer,
                task_dataloader=None,
                realignment_dataloader=realignment_dataloader,
                task_accumulation_steps=accumulation_steps,
                logging_steps=logging_steps,
                log_in_wandb=log_in_wandb,
                nb_iter=len(task_dataloader),
                realignment_coef=realignment_coef
                if realignment_coef_scheduler is None
                else realignment_coef_scheduler(i),
            )

    for i in range(n_epochs):

        epoch_loop(
            model,
            optimizer,
            task_dataloader=task_dataloader,
            realignment_dataloader=realignment_dataloader if strategy == "during" else None,
            task_accumulation_steps=accumulation_steps,
            logging_steps=logging_steps,
            log_in_wandb=log_in_wandb,
            realignment_coef=realignment_coef
            if realignment_coef_scheduler is None
            else realignment_coef_scheduler(i),
        )

        if evaluation_datasets is not None:
            res = evaluate_several_token_classification(
                tokenizer,
                model,
                evaluation_datasets,
                batch_size=task_batch_size,
                prefixes=evaluation_prefixes,
                overall_prefix="eval",
                metric_fn=metric_fn,
                collator=data_collator,
            )
            logging.info(res)
            if log_in_wandb:
                wandb.log(res)
        if same_language_evaluation_dataloader is not None:
            res = evaluate_token_classification(
                model, same_language_evaluation_dataloader, prefix="eval_same", metric_fn=metric_fn
            )
            logging.info(res)
            if log_in_wandb:
                wandb.log(res)

    if strategy == "after":
        for i in range(n_epochs):
            epoch_loop(
                model,
                optimizer,
                task_dataloader=None,
                realignment_dataloader=realignment_dataloader,
                task_accumulation_steps=accumulation_steps,
                logging_steps=logging_steps,
                log_in_wandb=log_in_wandb,
                nb_iter=len(task_dataloader),
                realignment_coef=realignment_coef
                if realignment_coef_scheduler is None
                else realignment_coef_scheduler(i),
            )

            if evaluation_datasets is not None:
                res = evaluate_several_token_classification(
                    tokenizer,
                    model,
                    evaluation_datasets,
                    batch_size=task_batch_size,
                    prefixes=evaluation_prefixes,
                    overall_prefix="eval",
                    metric_fn=metric_fn,
                    collator=data_collator,
                )
                logging.info(res)
                if log_in_wandb:
                    wandb.log(res)
            if same_language_evaluation_dataloader is not None:
                res = evaluate_token_classification(
                    model,
                    same_language_evaluation_dataloader,
                    prefix="eval_same",
                    metric_fn=metric_fn,
                )
                logging.info(res)
                if log_in_wandb:
                    wandb.log(res)

    if evaluation_datasets is not None:
        res = evaluate_several_token_classification(
            tokenizer,
            model,
            evaluation_datasets,
            batch_size=task_batch_size,
            prefixes=evaluation_prefixes,
            overall_prefix="final_eval",
            metric_fn=metric_fn,
            collator=data_collator,
        )
        logging.info(res)
        if log_in_wandb:
            wandb.log(res)
    if same_language_evaluation_dataloader is not None:
        res = evaluate_token_classification(
            model,
            same_language_evaluation_dataloader,
            prefix="final_eval_same",
            metric_fn=metric_fn,
        )
        logging.info(res)
        if log_in_wandb:
            wandb.log(res)
