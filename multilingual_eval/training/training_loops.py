import logging
from torch.utils.data import DataLoader, DistributedSampler
import torch
from transformers import DataCollatorForTokenClassification
from transformers.optimization import get_scheduler
from torch.optim import Adam
import numpy as np
import random
import math
import hashlib
import os
import json
import dataclasses

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
    learning_rate=2e-5,
    n_epochs=10,
    accumulation_steps=1,
    logging_steps=100,
    log_in_wandb=False,
    result_store=None,
    metric_fn=None,
    realignment_coef=0.1,
    realignment_coef_scheduler=None,
    data_collator=None,
    seed=None,
    epoch_callbacks=None,
    realignment_step_callbacks=None,
    hash_args=None,
    cache_dir=None,
    return_model_hash=False,
    final_prefix="final",
    pretrained_model_fn=None,
    realignment_steps_by_finetuning=1,
    label_key="labels",
    model_name=None
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
    - learning_rate: learning rate for both fine-tuning and realignment
    - n_epochs: number of epochs for the fine-tuning task
    - accumulation_steps: number of accumulation steps for the fine-tuning task
    - logging_steps: int, default 100, number of steps (in term of optimization steps, hence nb of batch / accumulation steps) between each log of training stats
    - log_in_wandb: (deprecated) whether to log training stats in wandb (conditional import), better to use result_store with loggers.WandbResultStore
    - result_store: an instance of loggers.DefaultResultStore that captures the different results obtained along the training (by default it logs them to the console, but it can store them in
        a dictionary for later retrieval)
    - metric_fn: function that gets the metric from the overall predictions and labels
    - realignment_coef: float, default 0.1, the coefficient to apply to the realignment loss
    - realignment_coef_scheduler: a function that takes an integer (the epoch) and return a float, the coefficient to apply to the realignment loss at
        given epochs, overrides realignment_coef
    - data_collator: default None, if None, will default to DataCollatorForTokenClassification(tokenizer)
    - seed
    - epoch_callbacks: (deprecated) optional list of function that takes the model as input and will be called before the first fine-tuning epoch and after each one
    - realignment_step_callbacks: (deprecated) like epoch_callbacks but for each realignment step
    - hash_args: (deprecated) default None, optional string to add to hashing realigned models (only with before strategy), will cache only if it is provided (ideally with model name, id for realignment dataset and commit hash)
        and if cache_dir is provided
    - cache_dir: default None, optional directory for caching models
    - return_model_hash: default False, whether to return the model hash for model saved after realignment (not fine-tuning !!!) useful only if hash_args and cache_dir are specified and if strategy == "before"
    - final_prefix: prefix for metrics in the final evaluation
    - pretrained_model_fn: (deprecated) when the model is cached, function to instantiate the pretrained model from cache_path
    - realignment_steps_by_finetuning: number of realignment optimization steps to perform by fine-tuning steps (useful in 'during' strategy)
    - label_key: (deprecated) the key for the labels in the training input
    """

    # Define a function to log the status of the Embedding space and each RobertaLayer (frozen or unfrozen)
    def log_layer_status(model, model_name):
        if "roberta" in model_name:
            # Handle RoBERTa and XLM-RoBERTa models
            if any(param.requires_grad for param in model.roberta.embeddings.parameters()):
                logging.info("Embedding Space: Unfrozen")
            else:
                logging.info("Embedding Space: Frozen")
            
            logging.info("Word Embeddings: {}".format(
                model.roberta.embeddings.word_embeddings.weight.requires_grad)
            )
            logging.info("Position Embeddings: {}".format(
                model.roberta.embeddings.position_embeddings.weight.requires_grad)
            )
            logging.info("Token Type Embeddings: {}".format(
                model.roberta.embeddings.token_type_embeddings.weight.requires_grad)
            )
            logging.info("LayerNorm: {}".format(
                model.roberta.embeddings.LayerNorm.weight.requires_grad)
            )
            logging.info("Dropout: {}".format(
                any(p.requires_grad for p in model.roberta.embeddings.dropout.parameters())
            ))

            # Log the status of encoder layers
            layers = list(model.roberta.encoder.layer)
            for i, layer in enumerate(layers):
                if any(param.requires_grad for param in layer.parameters()):
                    logging.info(f"RobertaLayer {i}: Unfrozen")
                else:
                    logging.info(f"RobertaLayer {i}: Frozen")

        elif "distilbert" in model_name:
            # Handle DistilBERT model
            if any(param.requires_grad for param in model.distilbert.embeddings.parameters()):
                logging.info("Embedding Space: Unfrozen")
            else:
                logging.info("Embedding Space: Frozen")

            logging.info("Word Embeddings: {}".format(
                model.distilbert.embeddings.word_embeddings.weight.requires_grad)
            )
            logging.info("Position Embeddings: {}".format(
                model.distilbert.embeddings.position_embeddings.weight.requires_grad)
            )
            logging.info("LayerNorm: {}".format(
                model.distilbert.embeddings.LayerNorm.weight.requires_grad)
            )
            logging.info("Dropout: {}".format(
                model.distilbert.embeddings.dropout.p)
            )

            # Log the status of transformer layers
            for i, layer in enumerate(model.distilbert.transformer.layer):
                if any(param.requires_grad for param in layer.parameters()):
                    logging.info(f"TransformerBlock {i}: Unfrozen")
                else:
                    logging.info(f"TransformerBlock {i}: Frozen")

        else:
            logging.warning(f"Model type '{model_name}' not recognized. No logging performed.")

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

    print()
    print(f'RUNNING MODEL: {model_name}')
    print(model)
    print()

    # If needed, create dataloader for re-alignment task
    if strategy.find('baseline') == -1:
        # Note: if this line is modified, hashing args for caching must be checked

        print("Not Using Baseline")
        # print('Realignment Dataset')
        # print(realignment_dataset)
        # print()

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
        print("Using Baseline")
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

    use_caching = False

    # If strategy is "before" or "before+during", perform realignment before fine-tuning
    if strategy in ["before", "before+during", 
                    "freeze_realign_unfreeze",
                    "freeze_realign_unfreeze_last_half",
                    "freeze_realign_unfreeze_last_6",
                   ]:
        use_caching = cache_dir is not None and hash_args is not None and seed is not None

        learning_rate = learning_rate

        realignment_steps_before = (
            math.ceil(len(task_dataloader) / accumulation_steps) * n_epochs
            if nb_realignment_steps_before is None
            else nb_realignment_steps_before
        )

        if use_caching:
            string_to_hash = (
                hash_args
                + "__"
                + "__".join(
                    [
                        str(learning_rate),
                        str(seed),
                        str(realignment_batch_size),
                        str(realignment_steps_before),
                    ]
                )
            )
            model_hash = hashlib.md5(string_to_hash.encode()).hexdigest()

            cache_path = os.path.join(cache_dir, model_hash)
            training_state_path = os.path.join(cache_dir, f"{model_hash}.json")
            info_path = os.path.join(cache_dir, f"{model_hash}.info")
        else:
            cache_path = None
            training_state_path = None
            info_path = None

        # Note: if this line is modified, hashing args for caching must be checked
        before_optimizer = Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)

        found_caching = False

        if cache_path is not None and os.path.isfile(training_state_path):
            try:
                with open(training_state_path, "r") as f:
                    other_training_state = TrainingState(**json.load(f))
                    training_state.update_from_other_finetuning(other_training_state)
                found_caching = True
            except json.decoder.JSONDecodeError:
                logging.error(f"Could not decode cached training state. Will not use the cache.")

        if found_caching:
            logging.info(f"Loading cached model: {model_hash}")
            model = (
                pretrained_model_fn(cache_path, ignore_mismatched_sizes=True)
                if pretrained_model_fn is not None
                else model.__class__.from_pretrained(cache_path, ignore_mismatched_sizes=True)
            )
        else:

            # print()
            # print('Realignment Dataloader')
            # for i, batch in enumerate(realignment_dataloader):
            #     # print(batch.keys())
            #     if i == 1:  # This will print the first batch
            #         break

            # print()
            # print('Fine-tuning Dataloader')
            # for i, batch in enumerate(task_dataloader):
            #     # print(batch)
            #     if i == 1:  # This will print the first batch
            #         break
            # print()

            print('')
            print('STARTING REALIGNMENT')
            print()

            if strategy == "freeze_realign_unfreeze" and "roberta" in model_name:
                print('Freezing first 6 encoders...')
                for i in range(6):
                    for param in model.roberta.encoder.layer[i].parameters():
                        param.requires_grad = False

                print('Freezing done...')

            if strategy == "freeze_realign_unfreeze_last_6" and "roberta" in model_name:
                freeze_realign_unfreeze_layers = 6
                print(f'Freezing last {freeze_realign_unfreeze_layers} encoders...')
                
                total_layers = len(model.roberta.encoder.layer)
                for i in range(total_layers - freeze_realign_unfreeze_layers, total_layers):
                    for param in model.roberta.encoder.layer[i].parameters():
                        param.requires_grad = False

                print('Freezing done...')

            if strategy == "freeze_realign_unfreeze" and "distilbert" in model_name:
                num_layers = len(model.distilbert.transformer.layer)
                layers_to_freeze = num_layers // 2  # Freezing the first half of the layers

                print(f'Freezing first {layers_to_freeze} transformer blocks...')
                for i in range(layers_to_freeze):
                    for param in model.distilbert.transformer.layer[i].parameters():
                        param.requires_grad = False

                print('Freezing done...')

            if strategy == "freeze_realign_unfreeze_last_half" and "distilbert" in model_name:
                num_layers = len(model.distilbert.transformer.layer)
                layers_to_freeze = num_layers // 2  # Number of layers to freeze

                # Calculate the starting index for freezing (freezing the last half of the layers)
                start_freezing_from_layer = num_layers - layers_to_freeze

                print(f'Freezing last {layers_to_freeze} transformer blocks...')
                for i in range(start_freezing_from_layer, num_layers):
                    for param in model.distilbert.transformer.layer[i].parameters():
                        param.requires_grad = False

                print('Freezing done...')

            log_layer_status(model, model_name)

            training_state = epoch_loop(
                model,
                before_optimizer,
                task_dataloader=None,
                realignment_dataloader=realignment_dataloader,
                task_accumulation_steps=1,
                logging_steps=logging_steps,
                log_in_wandb=log_in_wandb,
                result_store=result_store,
                nb_iter=realignment_steps_before,
                realignment_step_callbacks=realignment_step_callbacks,
                training_state=training_state,
                log_first_sample=True,
                realignment_steps_by_finetuning=realignment_steps_by_finetuning,
            )

            res = training_state.log_state()
            if log_in_wandb:
                wandb.log(res)
            if result_store:
                result_store.log(res)

            if cache_path is not None:
                logging.info(f"Saving realigned model: {model_hash}")
                model.save_pretrained(cache_path)

                with open(os.path.join(cache_path, "info.txt"), "w") as f:
                    f.write(string_to_hash + "\n")

                with open(training_state_path, "w") as f:
                    json.dump(dataclasses.asdict(training_state), f)

                with open(info_path, "w") as f:
                    f.write(hash_args + "\n")

            print('')
            print('DONE REALIGNMENT')
            print()

            if strategy == "freeze_realign_unfreeze" and "roberta" in model_name:
                print('Unfreezing first 6 encoders...')
                for i in range(6):
                    for param in model.roberta.encoder.layer[i].parameters():
                        param.requires_grad = True

                print('Unfreezing done...')

            
            if strategy == "freeze_realign_unfreeze_last_6" and "roberta" in model_name:
                print('Unfreezing last 6 encoders...')
                
                total_layers = len(model.roberta.encoder.layer)
                for i in range(total_layers - 6, total_layers):
                    for param in model.roberta.encoder.layer[i].parameters():
                        param.requires_grad = True

                print('Unfreezing done...')

            if strategy == "freeze_realign_unfreeze" and "distilbert" in model_name:
                num_layers = len(model.distilbert.transformer.layer)
                layers_to_freeze = num_layers // 2  # Freezing the first half of the layers

                print(f'Unfreezing first {layers_to_freeze} transformer blocks...')
                for i in range(layers_to_freeze):
                    for param in model.distilbert.transformer.layer[i].parameters():
                        param.requires_grad = True

                print('Unfreezing done...')

            if strategy == "freeze_realign_unfreeze_last_half" and "distilbert" in model_name:
                num_layers = len(model.distilbert.transformer.layer)
                layers_to_freeze = num_layers // 2  # Number of layers to freeze

                # Calculate the starting index for freezing (freezing the last half of the layers)
                start_freezing_from_layer = num_layers - layers_to_freeze

                print(f'Unfreezing last {layers_to_freeze} transformer blocks...')
                for i in range(start_freezing_from_layer, num_layers):
                    for param in model.distilbert.transformer.layer[i].parameters():
                        param.requires_grad = True

                print('Unfreezing done...')

    optimizer = Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
    scheduler = get_scheduler(
        "linear",
        optimizer,
        num_warmup_steps=int(0.1 * len(task_dataloader) * n_epochs),
        num_training_steps=len(task_dataloader) * n_epochs,
    )

    for callback in epoch_callbacks:
        callback(model)

    print()
    print('STARTING FINETUNING')
    print()

    if strategy in ["during_freeze_realign_unfreeze", 
                    "baseline_freeze_realign_unfreeze"] and "roberta" in model_name:
        print('Freezing first 6 encoders...')
        for i in range(6):
            for param in model.roberta.encoder.layer[i].parameters():
                param.requires_grad = False

        print('Freezing done...')

    if strategy in ["baseline_freeze_realign_unfreeze_last_6",
                    "during_freeze_realign_unfreeze_last_6"] and "roberta" in model_name:
        total_layers = len(model.roberta.encoder.layer)
        for i in range(total_layers - 6, total_layers):
            for param in model.roberta.encoder.layer[i].parameters():
                param.requires_grad = False

    if strategy == "during_freeze_realign_unfreeze" and "distilbert" in model_name:
        num_layers = len(model.distilbert.transformer.layer)
        layers_to_freeze = num_layers // 2  # Freezing the first half of the layers

        print(f'Freezing first {layers_to_freeze} transformer blocks...')
        for i in range(layers_to_freeze):
            for param in model.distilbert.transformer.layer[i].parameters():
                param.requires_grad = False

        print('Freezing done...')

    if strategy == "during_freeze_realign_unfreeze_last_half" and "distilbert" in model_name:
        num_layers = len(model.distilbert.transformer.layer)
        layers_to_freeze = num_layers // 2  # Number of layers to freeze

        # Calculate the starting index for freezing (freezing the last half of the layers)
        start_freezing_from_layer = num_layers - layers_to_freeze

        print(f'Freezing last {layers_to_freeze} transformer blocks...')
        for i in range(start_freezing_from_layer, num_layers):
            for param in model.distilbert.transformer.layer[i].parameters():
                param.requires_grad = False

        print('Freezing done...')
    
    realignment_optimizer = None
    realignment_scheduler = None
    if strategy.startswith("during_partial_freeze"):
        realigned_parameters = []

        if "roberta" in model_name:
            n_layers = len(model.roberta.encoder.layer)
            encoder_prefix = "roberta.encoder.layer"
        elif "distilbert" in model_name:
            n_layers = len(model.distilbert.transformer.layer)
            encoder_prefix = "distilbert.transformer.layer"
        else:
            raise NotImplementedError(f"during_partial_freeze_* strategies are not implemented for model {model}")
        
        if strategy.endswith("front"):
            prefixes_to_ignore = [f"{encoder_prefix}.{i}" for i in range(n_layers // 2)]
        elif strategy.endswith("back"):
            prefixes_to_ignore = [f"{encoder_prefix}.{i}" for i in range(n_layers // 2, n_layers)]
        elif strategy.endswith("none"):
            prefixes_to_ignore = []
        else:
            raise NotImplementedError(f"Unrecognized strategy {strategy}")
        
        for name, param in model.named_parameters():
            if any(map(lambda x: name.startswith(x), prefixes_to_ignore)):
                continue
            realigned_parameters.append(param)

        realignment_optimizer =  Adam(realigned_parameters, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
        realignment_scheduler = get_scheduler(
            "linear",
            realignment_optimizer,
            num_warmup_steps=int(0.1 * len(task_dataloader) * n_epochs),
            num_training_steps=len(task_dataloader) * n_epochs,
        )


    log_layer_status(model, model_name)

    for i in range(n_epochs):
        training_state = epoch_loop(
            model,
            optimizer,
            scheduler=scheduler,
            task_dataloader=task_dataloader,
            realignment_dataloader=realignment_dataloader
            if strategy in ["during",
                            "during_separate_backward",
                            "during_freeze_realign_unfreeze",
                            "during_freeze_realign_unfreeze_last_6",
                            "during_freeze_realign_unfreeze_last_half",
                            "before+during", 
                            "during_partial_freeze_front",
                            "during_partial_freeze_back",
                            "during_partial_freeze_none",
                            "staged"]
            else None,
            realignment_optimizer=realignment_optimizer,
            realignment_scheduler=realignment_scheduler,
            task_accumulation_steps=accumulation_steps,
            logging_steps=logging_steps,
            log_in_wandb=log_in_wandb,
            result_store=result_store,
            realignment_coef=realignment_coef
            if realignment_coef_scheduler is None
            else realignment_coef_scheduler(i),
            realignment_step_callbacks=realignment_step_callbacks,
            training_state=training_state,
            log_first_sample=i == 0,
            realignment_steps_by_finetuning=realignment_steps_by_finetuning,
            separate_backward=strategy == "during_separate_backward"
        )
        for callback in epoch_callbacks:
            callback(model)

        res = training_state.log_state()
        if log_in_wandb:
            wandb.log(res)
        if result_store:
            result_store.log(res)

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
                label_key=label_key,
            )
            logging.info(res)
            if log_in_wandb:
                wandb.log(res)
            if result_store:
                result_store.log(res)
        if same_language_evaluation_dataset is not None:
            res = evaluate_token_classification(
                model, same_language_evaluation_dataloader, prefix="eval_same", metric_fn=metric_fn
            )
            logging.info(res)
            if log_in_wandb:
                wandb.log(res)
            if result_store:
                result_store.log(res)

    print()
    print('DONE FINETUNING')
    print()

    if strategy == "after":
        after_optimizer = Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)

        training_state = epoch_loop(
            model,
            after_optimizer,
            task_dataloader=None,
            realignment_dataloader=realignment_dataloader,
            task_accumulation_steps=accumulation_steps,
            logging_steps=logging_steps,
            log_in_wandb=log_in_wandb,
            result_store=result_store,
            nb_iter=(
                len(task_dataloader) * n_epochs
                if nb_realignment_steps_before is None
                else nb_realignment_steps_before * accumulation_steps
            ),
            realignment_step_callbacks=realignment_step_callbacks,
            training_state=training_state,
            realignment_steps_by_finetuning=realignment_steps_by_finetuning,
        )
        res = training_state.log_state()
        if log_in_wandb:
            wandb.log(res)
        if result_store:
            result_store.log(res)
        for callback in epoch_callbacks:
            callback(model)

    if evaluation_datasets is not None:
        res = evaluate_several_token_classification(
            tokenizer,
            model,
            evaluation_datasets,
            batch_size=task_batch_size,
            prefixes=evaluation_prefixes,
            overall_prefix=f"{final_prefix}_eval",
            metric_fn=metric_fn,
            collator=data_collator,
            label_key=label_key,
        )
        logging.info(res)
        if log_in_wandb:
            wandb.log(res)
        if result_store:
            result_store.log(res)
    if same_language_evaluation_dataset is not None:
        res = evaluate_token_classification(
            model,
            same_language_evaluation_dataloader,
            prefix=f"{final_prefix}_eval_same",
            metric_fn=metric_fn,
        )
        logging.info(res)
        if log_in_wandb:
            wandb.log(res)
        if result_store:
            result_store.log(res)

    if return_model_hash and use_caching:
        return model_hash
