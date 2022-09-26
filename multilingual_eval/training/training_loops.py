import logging
from torch.utils.data import DataLoader, DistributedSampler
import torch
from transformers import DataCollatorForTokenClassification
from transformers.optimization import AdamW, get_scheduler


from multilingual_eval.training.epoch_loop import epoch_loop
from multilingual_eval.datasets.realignment_dataset import (
    RealignmentAndOtherCollator,
)
from multilingual_eval.training.evaluation_loops import (
    evaluate_several_token_classification,
    evaluate_token_classification,
)


def realignment_training_loop(
    output_dir,
    tokenizer,
    model,
    task_dataset: DataLoader,
    realignment_dataset: DataLoader,
    evaluation_datasets=None,
    same_language_evaluation_dataset=None,
    evaluation_prefixes=None,
    task_batch_size=4,
    realignment_batch_size=2,
    n_epochs=10,
    accumulation_steps=1,
    logging_steps=100,
    log_in_wandb=False,
    strategy="during",
    metric_fn=None,
    realignment_coef=0.1,
    realignment_coef_scheduler=None,
):
    if log_in_wandb:
        import wandb

    if model.device.type != "cuda" and torch.cuda.device_count() > 0:
        model = model.to(0)

    # define optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.999), eps=1e-8)

    task_dataloader = DataLoader(
        task_dataset,
        shuffle=True,
        batch_size=task_batch_size,
        collate_fn=DataCollatorForTokenClassification(tokenizer),
    )
    if strategy != "baseline":
        realignment_dataloader = DataLoader(
            realignment_dataset,
            shuffle=False,
            batch_size=realignment_batch_size,
            collate_fn=RealignmentAndOtherCollator(
                tokenizer,
                DataCollatorForTokenClassification(tokenizer),
            ),
        )
    else:
        realignment_dataloader = None

    if same_language_evaluation_dataset is not None:
        same_language_evaluation_dataloader = DataLoader(
            same_language_evaluation_dataset,
            shuffle=False,
            batch_size=task_batch_size,
            collate_fn=DataCollatorForTokenClassification(tokenizer),
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
