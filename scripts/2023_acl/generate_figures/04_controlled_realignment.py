"""
Code to generate results for a given model for experiments 21/22/23
given the CSV exported from wandb
"""

import pandas as pd
import sys, os
import numpy as np

sys.path.append(os.curdir)

from multilingual_eval.plotting_utils import get_statistics
from multilingual_eval.seeds import seeds

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file")
    parser.add_argument("task")
    parser.add_argument("--metric", type=str, default="accuracy")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--langs", nargs="+", default=["same", "ar", "es", "fr", "ru", "zh"])
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)
    if "task" in df.columns:
        df = df[df.task == args.task]
    df = df[
        (df.seed == seeds[0])
        | (df.seed == seeds[1])
        | (df.seed == seeds[2])
        | (df.seed == seeds[3])
        | (df.seed == seeds[4])
    ]

    langs = args.langs

    def get_strategy_name(strategy):
        if strategy == "baseline":
            return "only fine-tuned"
        parts = strategy.split("_")
        if len(parts) != 2:
            return strategy
        method, aligner = parts
        if method == "before":
            return f" before ({aligner})"
        if method == "during":
            return f" jointly ({aligner})"
        return strategy

    strategies = [
        "baseline",
        *[
            f"{strategy}_{aligner}"
            for strategy in ["before", "during"]
            for aligner in ["fastalign", "awesome", "dico"]
        ],
    ]
    strategy_to_rank = {s: i for i, s in enumerate(strategies)}

    existing_models = {
        "distilbert-base-multilingual-cased": "distilmBERT",
        "bert-base-multilingual-cased": "mBERT",
        "xlm-roberta-base": "XLM-R base",
        "xlm-roberta-large": "XLM-R large",
    }
    model_to_rank = {
        "distilbert-base-multilingual-cased": 0,
        "bert-base-multilingual-cased": 1,
        "xlm-roberta-base": 2,
        "xlm-roberta-large": 3,
    }

    if "model" in df.columns:
        models = df["model"].unique()
        print(models)
    else:
        models = [args.model]
        assert args.model is not None

    print(df.columns)

    res = ""

    res += " & ".join(args.langs) + "\\\\\n"
    res += "\\hline\n"

    for model in sorted(models, key=model_to_rank.__getitem__):

        if "model" in df.columns:
            subdf = df[df.model == model]
        else:
            subdf = df

        model_name = existing_models.get(model, model)

        if "method" in df.columns:
            strategies = subdf["method"].unique()
        else:
            strategies = subdf["realignment_strategy"].unique()

        means = np.zeros((len(langs), len(strategies)))
        stds = np.zeros((len(langs), len(strategies)))

        for j, strategy in enumerate(sorted(strategies, key=lambda x: strategy_to_rank.get(x, 10))):
            if "method" in df.columns:
                df_by_strategy = subdf[(subdf.method == strategy)]
            else:
                df_by_strategy = subdf[(subdf.realignment_strategy == strategy)]
            for i, lang in enumerate(langs):
                scores = list(df_by_strategy[f"final_eval_{lang}_{args.metric}"])
                means[i, j] = np.mean(scores)
                stds[i, j] = np.std(scores)

        for j, strategy in enumerate(sorted(strategies, key=lambda x: strategy_to_rank.get(x, 10))):
            res += model_name + " " + get_strategy_name(strategy)
            for i, lang in enumerate(langs):
                mean_argmax = np.argmax([0 if np.isnan(m) else m for m in means[i]])
                is_max = j == mean_argmax
                is_significant = means[i, j] - means[i, 0] > stds[i, 0]
                is_significantly_lower = means[i, 0] - means[i, j] > stds[i, 0] and j != 0
                is_not_significant = not (is_significant or is_significantly_lower) and j != 0
                res += (
                    " & "
                    + (
                        "\cellcolor{gray!30}"
                        if is_not_significant
                        else ("\cellcolor{gray!80}" if is_significantly_lower else "")
                    )
                    + ("\\textbf{" if is_max else "")
                    + f"{means[i,j]*100:.1f}"
                    + ("}" if is_max else "")
                    + f"$_{{\pm {stds[i,j]*100:.1f}}}$"
                )
            res += " \\\\\n"

        res += "\\hline\n"

    if args.output_file is not None:
        with open(args.output_file, "w") as f:
            f.write(res)

    print(res)
