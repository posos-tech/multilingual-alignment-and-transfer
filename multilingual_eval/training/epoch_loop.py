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
            task_loss = 0

            realignment_iterator, realignment_batch = get_next_or_restart(
                realignment_dataloader, realignment_iterator
            )

            realignment_loss = get_realignment_loss_for_batch(model, realignment_batch).loss

        if torch.cuda.device_count() > 1:
            outputs = torch.nn.parallel.data_parallel(model, None, module_kwargs=batch)
            tmp_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            task_loss += tmp_loss.mean()
        else:
            batch = bring_batch_to_model(batch, model)
            outputs = model(**batch)
            task_loss += outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        accumulated_steps += 1

        if i % task_accumulation_steps == task_accumulation_steps - 1:
            task_loss /= accumulated_steps

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
    if i % task_accumulation_steps != task_accumulation_steps - 1:
        task_loss /= accumulated_steps
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
