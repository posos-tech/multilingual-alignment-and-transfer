import logging
from torch.utils.data import DataLoader, DistributedSampler
import torch
from transformers import DataCollatorForTokenClassification
from transformers.optimization import get_scheduler
from torch.optim import Adam
import numpy as np
import random
import math

from multilingual_eval.training.states import TrainingState
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
    nb_realignment_steps_before=None,
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
    epoch_callbacks=None,
    realignment_step_callbacks=None,
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
    - nb_realignment_steps_before: if set, number of realignment batches to see before fine-tuning, otherwise it is n_epochs times the number of fine-tuning batch.
        Only taken into account if strategy is "before", "before+during" or "after"
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
    epoch_callbacks = epoch_callbacks or []
    realignment_step_callbacks = realignment_step_callbacks or []

    if log_in_wandb:
        import wandb

    # Put model to GPU if available
    if model.device.type != "cuda" and torch.cuda.device_count() > 0:
        model = model.to(0)

    # Fix random seed for Pytorch and numpy
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

    # Create dataloader for the fine-tuning task
    task_dataloader = DataLoader(
        task_dataset,
        shuffle=True,
        batch_size=task_batch_size,
        collate_fn=data_collator,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # If needed, create dataloader for re-alignment task
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

    training_state = TrainingState.compute_expected_samples(
        strategy,
        task_dataset,
        task_dataloader,
        n_epochs,
        task_batch_size,
        realignment_batch_size,
        accumulation_steps=accumulation_steps,
        nb_realignment_steps_before=nb_realignment_steps_before,
    )

    # If available, create dataloader for evaluation on training language
    if same_language_evaluation_dataset is not None:
        same_language_evaluation_dataloader = DataLoader(
            same_language_evaluation_dataset,
            shuffle=False,
            batch_size=task_batch_size,
            collate_fn=data_collator,
        )

    # If strategy is "before" or "before+during", perform realignment before fine-tuning
    if strategy in ["before", "before+during"]:

        before_optimizer = Adam(model.parameters(), lr=2e-5, betas=(0.9, 0.999), eps=1e-8)

        training_state = epoch_loop(
            model,
            before_optimizer,
            task_dataloader=None,
            realignment_dataloader=realignment_dataloader,
            task_accumulation_steps=accumulation_steps,
            logging_steps=logging_steps,
            log_in_wandb=log_in_wandb,
            nb_iter=(
                len(task_dataloader) * n_epochs
                if nb_realignment_steps_before is None
                else nb_realignment_steps_before * accumulation_steps
            ),
            realignment_step_callbacks=realignment_step_callbacks,
            training_state=training_state,
        )

        res = training_state.log_state()
        if log_in_wandb:
            wandb.log(res)

    optimizer = Adam(model.parameters(), lr=2e-5, betas=(0.9, 0.999), eps=1e-8)
    scheduler = get_scheduler(
        "linear",
        optimizer,
        num_warmup_steps=int(0.1 * len(task_dataloader) * 5),
        num_training_steps=len(task_dataloader) * 5,
    )

    for callback in epoch_callbacks:
        callback(model)

    for i in range(n_epochs):

        training_state = epoch_loop(
            model,
            optimizer,
            scheduler=scheduler,
            task_dataloader=task_dataloader,
            realignment_dataloader=realignment_dataloader
            if strategy in ["during", "before+during"]
            else None,
            task_accumulation_steps=accumulation_steps,
            logging_steps=logging_steps,
            log_in_wandb=log_in_wandb,
            realignment_coef=realignment_coef
            if realignment_coef_scheduler is None
            else realignment_coef_scheduler(i),
            realignment_step_callbacks=realignment_step_callbacks,
            training_state=training_state,
        )
        for callback in epoch_callbacks:
            callback(model)

        res = training_state.log_state()
        if log_in_wandb:
            wandb.log(res)

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
        after_optimizer = Adam(model.parameters(), lr=2e-5, betas=(0.9, 0.999), eps=1e-8)

        training_state = epoch_loop(
            model,
            after_optimizer,
            task_dataloader=None,
            realignment_dataloader=realignment_dataloader,
            task_accumulation_steps=accumulation_steps,
            logging_steps=logging_steps,
            log_in_wandb=log_in_wandb,
            nb_iter=(
                len(task_dataloader) * n_epochs
                if nb_realignment_steps_before is None
                else nb_realignment_steps_before * accumulation_steps
            ),
            realignment_step_callbacks=realignment_step_callbacks,
            training_state=training_state,
        )
        res = training_state.log_state()
        if log_in_wandb:
            wandb.log(res)
        for callback in epoch_callbacks:
            callback(model)

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
