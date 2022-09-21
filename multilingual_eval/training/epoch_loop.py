import itertools
import torch
import logging
import math
from multilingual_eval.training.parallelism import (
    gather_if_needed,
    parallel_apply_if_needed,
    replicate_if_possible,
    scatter_kwargs_if_needed,
)
from multilingual_eval.training.utils import bring_batch_to_model, get_next_or_restart


def get_realignment_loss_for_batch(model, realignment_batch):
    realignment_batch = bring_batch_to_model(realignment_batch, model)
    return model(**realignment_batch, return_dict=True)


def epoch_loop(
    model,
    optimizer,
    task_dataloader=None,
    realignment_dataloader=None,
    task_accumulation_steps=1,
    logging_steps=100,
    log_in_wandb=False,
    nb_iter=None,
):
    if realignment_dataloader is None and task_dataloader is None:
        raise Exception(
            "Both task_dataloader and realignment_dataloader cannot be None, we need to train on at least one dataloader"
        )

    if task_dataloader is None and nb_iter is None:
        raise Exception(
            f"If task_dataloader is not provided (got {task_dataloader}), you should provide nb_iter (got {nb_iter})"
        )

    if nb_iter is not None and task_dataloader is not None:
        logging.warning(
            f"nb_iter was provided ({nb_iter}) but so was task_dataloader. nb_iter will be ignored."
        )

    if task_dataloader is not None:
        nb_iter = len(task_dataloader)

    model.train()
    if log_in_wandb:
        import wandb

    if realignment_dataloader is not None:
        realignment_iterator = iter(realignment_dataloader)

    nb_batch = math.ceil(nb_iter / task_accumulation_steps)

    for i, batch in (
        enumerate(task_dataloader)
        if task_dataloader is not None
        else enumerate(itertools.repeat(None, nb_iter))
    ):
        if i % task_accumulation_steps == 0:
            optimizer.zero_grad()
            accumulated_steps = 0
            total_loss = 0
            task_loss = 0
            realignment_loss = 0

            if realignment_dataloader is not None:
                realignment_iterator, realignment_batch = get_next_or_restart(
                    realignment_dataloader, realignment_iterator
                )

                realignment_loss = get_realignment_loss_for_batch(model, realignment_batch).loss

        if batch is not None:
            if torch.cuda.device_count() > 1:
                outputs = torch.nn.parallel.data_parallel(model, None, module_kwargs=batch)
                tmp_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                task_loss += tmp_loss.mean()
            else:
                batch = bring_batch_to_model(batch, model)
                outputs = model(**batch)
                task_loss += outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            accumulated_steps += 1

        if i % task_accumulation_steps == task_accumulation_steps - 1 or i == nb_iter - 1:
            task_loss /= max(1, accumulated_steps)

            # Note that the coefficient is already in the model definition
            total_loss = task_loss + realignment_loss

            total_loss.backward()

            optimizer.step()

            if logging_steps is not None and (i // task_accumulation_steps) % logging_steps == 0:
                batch_seen = math.ceil(i / task_accumulation_steps)

                logging.info(f"batch: {batch_seen}/{nb_batch} loss : {total_loss}")
                if log_in_wandb:
                    wandb.log(
                        {
                            "train_step": batch_seen,
                            "train_loss": total_loss,
                            "realignment_loss": realignment_loss,
                            "task_loss": task_loss,
                        }
                    )
