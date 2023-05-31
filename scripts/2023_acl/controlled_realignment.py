import os
import sys
import logging
import datasets
from typing import List
from transformers import AutoTokenizer, set_seed

sys.path.append(os.curdir)

from multilingual_eval.seeds import seeds
from multilingual_eval.loggers import (
    WandbResultStore,
    DictResultStore,
    DefaultResultStore,
)
from multilingual_eval.tokenization.chinese_segmenter import StanfordSegmenter
from multilingual_eval.training.wandb_utils import (
    wrap_train,
    imitate_wandb_sweep,
    store_dicts_in_csv,
)
from multilingual_eval.training.training_loops import realignment_training_loop
from multilingual_eval.training.evaluation_loops import evaluate_xquad
from multilingual_eval.training.batch_sizes import get_batch_size
from multilingual_eval.datasets.dispatch_datasets import (
    get_dataset_fn,
    get_dataset_metric_fn,
    model_fn,
    collator_fn,
)
from multilingual_eval.datasets.realignment_task import get_multilingual_realignment_dataset


def train(
    left_lang: str,
    right_langs: List[str],
    translation_dir: str,
    fastalign_dir: str,
    dico_dir: str,
    awesome_dir: str,
    config=None,
    sweep_config=None,
    zh_segmenter=None,
    debug=False,
    cache_dir=None,
    large_gpu=False,
    n_epochs=5,
    layers=None,
    result_store=None,
):
    layers = layers or [-1]
    model_name = config["model"]
    task_name = config["task"]
    seed = config["seed"]
    method = config["method"]
    if method == "baseline":
        aligner = None
    else:
        method, aligner = method.split("_")

    result_store = result_store or DefaultResultStore()

    if seed in seeds[5:]:
        logging.error(
            f"Not a seed we want to run, because we limit ourselves to 5, otherwise we won't have time to run everything"
        )
        return

    string_to_hash_for_caching = f"v1.0\nleft_lang={left_lang}\nright_langs={'-'.join(sorted(right_langs))}\nseed={seed}\nmodel_name={model_name}\ndebug={debug}"

    cumul_batch_size = 32
    batch_size = get_batch_size(model_name, cumul_batch_size, large_gpu=large_gpu)
    accumulation_steps = cumul_batch_size // batch_size

    realignment_batch_size = 16

    string_to_hash_for_caching += f"\nrealignment_batch_size={realignment_batch_size}"

    assert cumul_batch_size % batch_size == 0

    data_cache_dir = os.path.join(cache_dir, "datasets") if cache_dir is not None else cache_dir
    model_cache_dir = (
        os.path.join(cache_dir, "transformers") if cache_dir is not None else cache_dir
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache_dir)
    set_seed(seed)
    if method == "baseline":
        model = model_fn(task_name, with_realignment=False)(model_name, cache_dir=model_cache_dir)
    else:
        model = model_fn(task_name, with_realignment=True)(
            model_name,
            cache_dir=model_cache_dir,
            nb_pairs=len(right_langs),
            strong_alignment=True,
            realignment_loss="contrastive",
            with_mapping=False,
            regularization_to_init=False,
            realignment_layers=layers,
        )

    logging.debug(model)

    training_dataset = get_dataset_fn(task_name, zh_segmenter=zh_segmenter)(
        left_lang,
        tokenizer,
        split="train",
        limit=1000 if debug else None,
        datasets_cache_dir=data_cache_dir,
    )

    validation_datasets = get_dataset_fn(task_name, zh_segmenter=zh_segmenter)(
        right_langs,
        tokenizer,
        split="test",
        limit=100 if debug else None,
        datasets_cache_dir=data_cache_dir,
        interleave=False,
    )

    source_validation_dataset = get_dataset_fn(task_name, zh_segmenter=zh_segmenter)(
        left_lang,
        tokenizer,
        split="test",
        limit=100 if debug else None,
        datasets_cache_dir=data_cache_dir,
    )

    lang_pairs = [(left_lang, right_lang) for right_lang in right_langs]
    if aligner == "fastalign":
        alignment_dataset = get_multilingual_realignment_dataset(
            tokenizer, translation_dir, fastalign_dir, lang_pairs, max_length=96, seed=seed
        )
        string_to_hash_for_caching += f"\nalignment_dir={fastalign_dir}"
    elif aligner == "dico":
        alignment_dataset = get_multilingual_realignment_dataset(
            tokenizer, translation_dir, dico_dir, lang_pairs, max_length=96, seed=seed
        )
        string_to_hash_for_caching += f"\nalignment_dir={dico_dir}"
    elif aligner == "awesome":
        alignment_dataset = get_multilingual_realignment_dataset(
            tokenizer, translation_dir, awesome_dir, lang_pairs, max_length=96, seed=seed
        )
        string_to_hash_for_caching += f"\nalignment_dir={awesome_dir}"
    elif aligner is None:
        alignment_dataset = None
    else:
        raise KeyError(aligner)

    realignment_training_loop(
        tokenizer,
        model,
        training_dataset,
        alignment_dataset,
        strategy=method,
        evaluation_datasets=validation_datasets if task_name not in ["xquad"] else None,
        same_language_evaluation_dataset=source_validation_dataset
        if task_name not in ["xquad"]
        else None,
        evaluation_prefixes=right_langs,
        seed=seed,
        task_batch_size=batch_size,
        learning_rate=7.5e-6 if "roberta" in model_name else 2e-5,
        realignment_batch_size=realignment_batch_size,
        realignment_steps_by_finetuning=1,
        n_epochs=n_epochs,
        accumulation_steps=accumulation_steps,
        result_store=result_store,
        metric_fn=get_dataset_metric_fn(task_name)(),
        data_collator=collator_fn(task_name)(tokenizer),
    )

    if task_name in ["xquad"]:
        evaluate_xquad(
            model,
            tokenizer,
            left_lang,
            right_langs,
            batch_size=batch_size,
            debug=debug,
            data_cache_dir=data_cache_dir,
            result_store=result_store,
        )


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    default_strategies = [
        "baseline",
        *[
            f"{strategy}_{aligner}"
            for strategy in ["during", "before"]
            for aligner in ["fastalign", "dico", "awesome"]
        ],
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--translation_dir",
        type=str,
        default=None,
        help="Directory where the parallel dataset can be found, must be set if other strategy than baseline is used.",
    )
    parser.add_argument(
        "--fastalign_dir",
        type=str,
        default=None,
        help="Directory where fastalign alignments can be found, must be set if strategy ending in _fastalign is used",
    )
    parser.add_argument(
        "--dico_dir",
        type=str,
        help="Directory where bilingual dictionary alignments can be found, must be set if strategy ending in _dico is used",
    )
    parser.add_argument(
        "--awesome_dir",
        type=str,
        help="Directory where awesome alignments can be found, must be set if strategy ending in awesome is used",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=[
            "bert-base-multilingual-cased",
            "xlm-roberta-base",
            "distilbert-base-multilingual-cased",
        ],
    )
    parser.add_argument("--tasks", nargs="+", type=str, default=["wikiann", "udpos", "xnli"])
    parser.add_argument("--strategies", nargs="+", type=str, default=default_strategies)
    parser.add_argument("--left_lang", type=str, default="en")
    parser.add_argument(
        "--right_langs", type=str, nargs="+", default=["ar", "es", "fr", "ru", "zh"]
    )
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--debug", action="store_true", dest="debug")
    parser.add_argument("--large_gpu", action="store_true", dest="large_gpu")
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--layers", type=int, default=[-1])
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true", dest="use_wandb")
    parser.set_defaults(debug=False, large_gpu=False, use_wandb=False)
    args = parser.parse_args()

    if not args.use_wandb and args.output_file is None:
        raise Exception(
            f"Either wandb must be used (--use_wandb) or an output csv file must be set (--output_file) to store results"
        )
    if not args.use_wandb:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    sweep_config = {
        "method": "grid",
        "parameters": {
            "seed": {"values": seeds[: args.n_seeds]},
            "model": {"values": args.models},
            "task": {"values": args.tasks},
            "method": {"values": args.strategies},
        },
    }

    if args.debug:
        sweep_config["parameters"]["seed"]["values"] = sweep_config["parameters"]["seed"]["values"][
            :1
        ]

    with StanfordSegmenter() as zh_segmenter:
        if args.use_wandb:
            import wandb

            result_store = WandbResultStore()

            if args.sweep_id is None:
                sweep_id = wandb.sweep(sweep_config, project="controlled_realignment")
            else:
                sweep_id = args.sweep_id

            final_train_fn = wrap_train(
                lambda cfg, sweep_cfg, zh_sgm: train(
                    args.left_lang,
                    args.right_langs,
                    args.translation_dir,
                    args.fastalign_dir,
                    args.dico_dir,
                    args.awesome_dir,
                    layers=args.layers,
                    config=cfg,
                    sweep_config=sweep_cfg,
                    zh_segmenter=zh_sgm,
                    debug=args.debug,
                    large_gpu=args.large_gpu,
                    cache_dir=args.cache_dir,
                    n_epochs=args.n_epochs,
                    result_store=result_store,
                ),
                sweep_config,
                sweep_id,
                zh_segmenter=zh_segmenter,
            )

            wandb.agent(sweep_id, final_train_fn, project="controlled_realignment")
        else:
            datasets.disable_progress_bar()
            results = []
            for run_config in imitate_wandb_sweep(sweep_config):
                result_store = DictResultStore()
                result_store.log(run_config)
                train(
                    args.left_lang,
                    args.right_langs,
                    args.translation_dir,
                    args.fastalign_dir,
                    args.dico_dir,
                    args.awesome_dir,
                    layers=args.layers,
                    config=run_config,
                    sweep_config=sweep_config,
                    zh_segmenter=zh_segmenter,
                    debug=args.debug,
                    large_gpu=args.large_gpu,
                    cache_dir=args.cache_dir,
                    n_epochs=args.n_epochs,
                    result_store=result_store,
                )
                results.append(result_store.get_results())

            store_dicts_in_csv(args.output_file, results)
