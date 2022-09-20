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


def during_strategy_epoch_loop(
    model,
    optimizer,
    task_dataloader,
    realignment_dataloader,
    task_accumulation_steps=1,
    logging_steps=100,
    log_in_wandb=False,
):
    model.train()
    if log_in_wandb:
        import wandb

    realignment_iterator = iter(realignment_dataloader)

    nb_batch = math.ceil(len(task_dataloader) / task_accumulation_steps)

    for i, batch in enumerate(task_dataloader):
        if i % task_accumulation_steps == 0:
            optimizer.zero_grad()
            accumulated_steps = 0
            total_loss = 0

            realignment_iterator, realignment_batch = get_next_or_restart(
                realignment_dataloader, realignment_iterator
            )

            # Note that the coefficient is already in the model definition
            realignment_loss = get_realignment_loss_for_batch(model, realignment_batch).loss
            replicas = replicate_if_possible(model)
            task_loss = [0] * len(replicas)
            max_replica_reached = 0

        batches = scatter_kwargs_if_needed(batch)
        outputs = parallel_apply_if_needed(replicas[: len(batches)], ((),) * len(batches), batches)
        for i, output in enumerate(outputs):
            task_loss[i] += output["loss"] if isinstance(output, dict) else output[0]
            max_replica_reached = max(i, max_replica_reached)
            accumulated_steps += 1

        if i % task_accumulation_steps == task_accumulation_steps - 1:
            task_loss = gather_if_needed(task_loss[:max_replica_reached], output_device=0)
            task_loss = sum(task_loss) / accumulated_steps

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
    if i % task_accumulation_steps != task_accumulation_steps - 1:
        task_loss = gather_if_needed(task_loss[:max_replica_reached], output_device=0)
        task_loss = sum(task_loss) / accumulated_steps

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
