"""
Code to generate results for a given model for experiments 21/22/23
given the CSV exported from wandb
"""

import pandas as pd
import sys, os
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes

sys.path.append(os.curdir)

from multilingual_eval.plotting_utils import get_statistics, COLOR_PALETTE
from multilingual_eval.seeds import seeds

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("csv_files", nargs="+")
    parser.add_argument("--tasks", type=str, nargs="+", default=None)
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=[
            "baseline",
            "before_fastalign",
            "before_awesome",
            "before_dico",
            "during_fastalign",
            "during_awesome",
            "during_dico",
        ],
    )
    args = parser.parse_args()

    dfs = [pd.read_csv(fname) for fname in args.csv_files]

    df = pd.concat(dfs)

    task_to_metric = defaultdict(lambda: "accuracy")
    task_to_metric["wikiann"] = "f1"

    task_to_name = {"wikiann": "NER", "udpos": "POS", "xnli": "NLI"}

    available_tasks = df["task"].unique()
    if args.tasks is None:
        tasks = list(available_tasks)
    else:
        tasks = sorted(list(set(available_tasks).intersection(args.tasks)), key=args.tasks.index)

    def get_strategy_name(model_name, strategy):
        if strategy == "baseline":
            return model_name
        parts = strategy.split("_")
        if len(parts) != 2:
            raise NotImplementedError(f"Unrecognized strategy: {strategy}")
        method, aligner = parts
        if method == "before":
            return f"+ before {aligner}"
        if method == "during":
            return f"+ joint {aligner}"
        raise NotImplementedError(f"Unrecognized strategy: {strategy}")

    strategies = args.strategies
    assert strategies[0] == "baseline"
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

    models = df["model"].unique()

    res = "& "

    res += " & ".join(map(task_to_name.__getitem__, tasks)) + "\\\\\n"
    res += "\\hline\n"

    for model in sorted(models, key=model_to_rank.__getitem__):

        subdf = df[df.model == model]

        model_name = existing_models.get(model, model)

        relevant_strategies = list(set(strategies).intersection(subdf["method"].unique()))

        means = np.zeros((len(relevant_strategies), len(tasks)))
        stds = np.zeros((len(relevant_strategies), len(tasks)))

        for i, strategy in enumerate(sorted(relevant_strategies, key=strategy_to_rank.__getitem__)):

            subsubdf = subdf[subdf.method == strategy]

            if len(subsubdf) == 0:
                continue

            for j, task in enumerate(tasks):
                subsubsubdf = subsubdf[subsubdf.task == task]

                scores = list(subsubsubdf[f"final_eval_avg_{task_to_metric[task]}"])

                mean = np.mean(scores)
                std = np.std(scores)

                means[i, j] = mean
                stds[i, j] = std

        for i, strategy in enumerate(sorted(relevant_strategies, key=strategy_to_rank.__getitem__)):

            res += f"{get_strategy_name(model_name, strategy)} "

            for j, task in enumerate(tasks):
                if i == 0:
                    res += f"& \\textit{{ {means[i,j]*100:.1f} }} "
                else:
                    res += "& "
                    if means[0, j] - means[i, j] > stds[0, j]:
                        res += "\\cellcolor{gray!80} "
                    elif not means[i, j] - means[0, j] > stds[0, j]:
                        res += "\\cellcolor{gray!30} "

                    if f"{means[i,j]*100:.1f}" == f"{max(means[:,j])*100:.1f}":
                        res += f"\\textbf{{ {means[i,j]*100:.1f} }} "
                    else:
                        res += f"{means[i,j]*100:.1f} "

            res += "\\\\\n"

        res += "\\hline\n"

    print(res)

    for task in tasks:
        print("\n\n")

        taskdf = df[df.task == task]

        res = f"{task_to_name[task]} & "

        langs = ["same", "ar", "es", "fr", "ru", "zh"]

        res += " & ".join(langs) + "\\\\\n"
        res += "\\hline\n"

        for model in sorted(models, key=model_to_rank.__getitem__):

            subdf = taskdf[taskdf.model == model]

            model_name = existing_models.get(model, model)

            strategies = subdf["method"].unique()

            means = np.zeros((len(langs), len(strategies)))
            stds = np.zeros((len(langs), len(strategies)))

            for j, strategy in enumerate(
                sorted(strategies, key=lambda x: strategy_to_rank.get(x, 10))
            ):
                if "method" in df.columns:
                    df_by_strategy = subdf[(subdf.method == strategy)]
                else:
                    df_by_strategy = subdf[(subdf.realignment_strategy == strategy)]
                for i, lang in enumerate(langs):
                    scores = list(df_by_strategy[f"final_eval_{lang}_{task_to_metric[task]}"])
                    means[i, j] = np.mean(scores)
                    stds[i, j] = np.std(scores)

            for j, strategy in enumerate(
                sorted(strategies, key=lambda x: strategy_to_rank.get(x, 10))
            ):
                res += get_strategy_name(model_name, strategy)
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
                        + ("\\textit{" if j == 0 else "")
                        + ("\\textbf{" if is_max else "")
                        + f"{means[i,j]*100:.1f}"
                        + ("}" if is_max else "")
                        + ("}" if j == 0 else "")
                        + f"$_{{\pm {stds[i,j]*100:.1f}}}$"
                    )
                res += " \\\\\n"

            res += "\\hline\n"

        print(res)

    model_sizes = {
        "distilbert-base-multilingual-cased": 66,
        "bert-base-multilingual-cased": 110,
        "xlm-roberta-base": 125,
        "xlm-roberta-large": 345,
    }

    summary_lang = "ar"
    summary_method = "before_dico"

    fig = plt.figure()
    print(len(tasks))
    print(available_tasks)

    gs = fig.add_gridspec(len(tasks), hspace=0)
    # _ = gs.subplots(sharex=True, sharey=True)
    for i, task in enumerate(tasks):
        ax = brokenaxes(xlims=((60 - 10, 125 + 10), (345 - 10, 345 + 8)), subplot_spec=gs[i])

        ax.set(ylim=[0.0, 0.85])

        ax.text(
            90,
            0.4,
            task_to_name[task],
            fontsize=14,
            horizontalalignment="center",
        )

        for j, (model, size) in enumerate(model_sizes.items()):
            if i == 0:
                ax.text(
                    size - 4 if model == "xlm-roberta-large" else size,
                    0.75,
                    existing_models[model],
                    fontsize=11,
                    horizontalalignment="center",
                    verticalalignment="bottom" if model == "xlm-roberta-base" else "top",
                )

            values_without = df[
                (df.model == model) & (df.task == task) & (df.method == "baseline")
            ][f"final_eval_{summary_lang}_{task_to_metric[task]}"].dropna()
            values_with = df[
                (df.model == model) & (df.task == task) & (df.method == summary_method)
            ][f"final_eval_{summary_lang}_{task_to_metric[task]}"].dropna()

            ax.bar(
                [size],
                [np.mean(values_without)],
                width=-7,
                align="edge",
                yerr=[np.std(values_without)],
                color=COLOR_PALETTE[0],
                **({"label": "without"} if i == len(tasks) - 1 and j == 0 else {}),
            )
            ax.bar(
                [size],
                [np.mean(values_with)],
                width=7,
                align="edge",
                yerr=[np.std(values_with)],
                color=COLOR_PALETTE[1],
                **({"label": "with"} if i == len(tasks) - 1 and j == 0 else {}),
            )

        ax.label_outer()

        ax.set_ylabel(task_to_metric[task])

        if i == len(tasks) - 1:
            ax.legend(loc="lower left")
            ax.set_xlabel("model size (million parameters)")

    plt.savefig("summary.pdf")
    plt.clf()
