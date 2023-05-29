import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import matplotlib.lines as mlines
import os
import sys
from transformers import AutoModel
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr, bootstrap
from pathlib import Path
import itertools

sys.path.append(os.curdir)

from multilingual_eval.plotting_utils import COLOR_PALETTE, spearman_with_bootstrap_ci
from multilingual_eval.utils import get_nb_layers


def get_scientific_notation_with_exponent(number, decimals=1):
    res = f"{number:.{decimals}E}"
    left, right = res.split("E")
    return f"{left}\\times 10^{{{right}}}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", type=str, nargs="+")
    parser.add_argument("--output_dir", type=str, default="local_runs/results")
    parser.add_argument("--langs", type=str, nargs="+", default=["ar", "es", "fr", "ru", "zh"])
    parser.add_argument(
        "--plots",
        type=str,
        nargs="+",
        default=[
            "bar_charts",
            "drop_table",
            "corr_table",
            "corr_table_by_seed",
            "correlation_by_layer_table",
            "correlation_charts",
            "before_vs_after",
            "before_vs_after_last",
        ],
    )
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()

    df = pd.concat([pd.read_csv(fname) for fname in args.csv_file])

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    models = {
        "distilbert-base-multilingual-cased": "distilmBERT",
        "bert-base-multilingual-cased": "mBERT",
        "xlm-roberta-base": "XLM-R Base",
        "xlm-roberta-large": "XLM-R Large",
    }

    model_in_df = set(df["model"].unique())

    models = {key: value for key, value in models.items() if key in model_in_df}

    tasks = df["task"].unique()

    task_to_name = {"wikiann": "NER", "udpos": "POS", "xnli": "NLI", "pawsx": "PAWS"}

    task_to_metric = defaultdict(lambda: "accuracy")
    task_to_metric["wikiann"] = "f1"

    langs = args.langs
    if not isinstance(langs, list):
        langs = list(langs)

    model_to_nlayer = {}

    for model_name in models:
        model = AutoModel.from_pretrained(model_name, cache_dir=args.cache_dir)
        n_layers = get_nb_layers(model)
        model_to_nlayer[model_name] = n_layers

    if "bar_charts" in args.plots:
        bar_chart_dir = os.path.join(args.output_dir, "bar_charts")

        Path(bar_chart_dir).mkdir(exist_ok=True)

        for model_slug in df["model"].unique():
            # bar chart with before-after for one language, one model
            model_name = models.get(model_slug, model_slug)

            model_bar_chart_dir = os.path.join(bar_chart_dir, model_slug)
            Path(model_bar_chart_dir).mkdir(exist_ok=True)

            for task in tasks:
                task_bar_chart_dir = os.path.join(model_bar_chart_dir, task)
                Path(task_bar_chart_dir).mkdir(exist_ok=True)

                bar_chart_df = df[(df.model == model_slug) & (df.task == task)]

                for direction in ["fwd", "bwd"]:
                    for lang in args.langs:
                        property_names_before = [
                            f"alignment_before_{direction}_{lang}_{i}"
                            for i in range(model_to_nlayer[model_slug])
                        ]
                        property_names_after = [
                            f"alignment_after_{direction}_{lang}_{i}"
                            for i in range(model_to_nlayer[model_slug])
                        ]
                        layers = list(range(model_to_nlayer[model_slug]))

                        property_names_after.sort(key=lambda x: int(x.split("_")[-1]) + 1)
                        property_names_before.sort(key=lambda x: int(x.split("_")[-1]) + 1)
                        layers.sort()

                        mean_after = [
                            np.mean(bar_chart_df[c].dropna()) for c in property_names_after
                        ]
                        mean_before = [
                            np.mean(bar_chart_df[c].dropna()) for c in property_names_before
                        ]
                        std_after = [np.std(bar_chart_df[c].dropna()) for c in property_names_after]
                        std_before = [
                            np.std(bar_chart_df[c].dropna()) for c in property_names_before
                        ]

                        plt.bar(
                            layers,
                            mean_before,
                            width=-0.4,
                            align="edge",
                            yerr=std_before,
                            color=COLOR_PALETTE[0],
                            label="before",
                        )
                        plt.bar(
                            layers,
                            mean_after,
                            width=0.4,
                            align="edge",
                            yerr=std_after,
                            color=COLOR_PALETTE[1],
                            label="after",
                        )

                        plt.legend(loc="upper right")
                        plt.xlabel("layer")
                        plt.ylabel("retrieval accuracy")
                        plt.savefig(os.path.join(task_bar_chart_dir, f"{lang}_{direction}.pdf"))
                        plt.clf()

    if "drop_table" in args.plots:
        # table with drop for last layer and all language
        res = "task & model & "

        res += " & ".join(map(lambda x: f"en-{x}", langs))
        res += "\\\\\n"
        res += "\\hline"

        for task in tasks:
            task_name = task_to_name[task]
            res += f"\\multirow{{{len(models)}}}{{*}}{{{task_name}}}"
            for model, model_name in models.items():
                if model not in df["model"].unique():
                    continue
                last_layer = model_to_nlayer[model] - 1
                res += f"& {model_name}"
                subdf = df[(df.model == model) & (df.task == task)]
                for lang in langs:
                    res += " & "
                    lang_drop = (
                        subdf[f"alignment_after_fwd_{lang}_{last_layer}"].dropna()
                        - subdf[f"alignment_before_fwd_{lang}_{last_layer}"].dropna()
                    ) / subdf[f"alignment_before_fwd_{lang}_{last_layer}"].dropna()
                    res += f" {np.mean(lang_drop):.2f}$_{{\pm {np.std(lang_drop):.2f}}}$"
                res += "\\\\\n"
            res += "\\hline"

        print(res)

        print("\n\n")

    if "corr_table" in args.plots:
        # Table by task and by moment

        res = "task & layer & \multicolumn{2}{|c}{en-X} & \multicolumn{2}{|c}{X-en}\\\\\n"
        res += "& & before & after & before & after \\\\\n\\hline\n"
        ci_res = "task & layer & \multicolumn{2}{|c}{en-X} & \multicolumn{2}{|c}{X-en}\\\\\n"
        ci_res += "& & before & after & before & after \\\\\n\\hline\n"

        models_for_layers = ["bert-base-multilingual-cased", "xlm-roberta-base"]

        for task in tasks:
            task_name = task_to_name[task]
            res += f"\\multirow{{2}}{{*}}{{{task_name}}}"
            ci_res += f"\\multirow{{2}}{{*}}{{{task_name}}}"

            alignment_scores_fwd_before = []
            alignment_scores_fwd_after = []
            alignment_scores_bwd_before = []
            alignment_scores_bwd_after = []
            delta_scores = []

            for model, n_layer in model_to_nlayer.items():
                if model not in df["model"].unique():
                    continue

                layer = n_layer - 1
                subdf = df[((df.model == model) & (df.task == task))]

                for lang in langs:
                    alignment_scores_fwd_before += list(
                        subdf[f"alignment_before_fwd_{lang}_{layer}"].dropna()
                    )
                    alignment_scores_fwd_after += list(
                        subdf[f"alignment_after_fwd_{lang}_{layer}"].dropna()
                    )
                    alignment_scores_bwd_before += list(
                        subdf[f"alignment_before_bwd_{lang}_{layer}"].dropna()
                    )
                    alignment_scores_bwd_after += list(
                        subdf[f"alignment_after_bwd_{lang}_{layer}"].dropna()
                    )
                    delta_scores += list(
                        (
                            subdf[f"final_eval_{lang}_{task_to_metric[task]}"]
                            - subdf[f"final_eval_same_{task_to_metric[task]}"]
                        )
                        / subdf[f"final_eval_same_{task_to_metric[task]}"]
                    )

            rho_fwd_before, p_fwd_before = spearmanr(alignment_scores_fwd_before, delta_scores)
            rho_fwd_after, p_fwd_after = spearmanr(alignment_scores_fwd_after, delta_scores)
            rho_bwd_before, p_bwd_before = spearmanr(alignment_scores_bwd_before, delta_scores)
            rho_bwd_after, p_bwd_after = spearmanr(alignment_scores_bwd_after, delta_scores)

            res += (
                " & last & "
                + ("\cellcolor{gray!30}" if p_fwd_before > 0.05 else "")
                + f"{rho_fwd_before:.2f} & "
                + ("\cellcolor{gray!30}" if p_fwd_after > 0.05 else "")
                + f"{rho_fwd_after:.2f} & "
                + ("\cellcolor{gray!30}" if p_bwd_before > 0.05 else "")
                + f"{rho_bwd_before:.2f} & "
                + ("\cellcolor{gray!30}" if p_bwd_after > 0.05 else "")
                + f"{rho_bwd_after:.2f} \\\\\n"
            )

            boostrap_fwd_before = spearman_with_bootstrap_ci(
                alignment_scores_fwd_before, delta_scores, n_resamples=2_000
            )
            boostrap_fwd_after = spearman_with_bootstrap_ci(
                alignment_scores_fwd_after, delta_scores, n_resamples=2_000
            )
            boostrap_bwd_before = spearman_with_bootstrap_ci(
                alignment_scores_bwd_before, delta_scores, n_resamples=2_000
            )
            boostrap_bwd_after = spearman_with_bootstrap_ci(
                alignment_scores_bwd_after, delta_scores, n_resamples=2_000
            )

            ci_res += (
                " & penult. "
                + f"& {rho_fwd_before:.2f} ({boostrap_fwd_before.confidence_interval.low:.2f} - "
                + f"{boostrap_fwd_before.confidence_interval.high:.2f}) "
                + f"& {rho_fwd_after:.2f} ({boostrap_fwd_after.confidence_interval.low:.2f} - "
                + f"{boostrap_fwd_after.confidence_interval.high:.2f}) "
                + f"& {rho_bwd_before:.2f} ({boostrap_bwd_before.confidence_interval.low:.2f} - "
                + f"{boostrap_bwd_before.confidence_interval.high:.2f}) "
                + f"& {rho_bwd_after:.2f} ({boostrap_bwd_after.confidence_interval.low:.2f} - "
                + f"{boostrap_bwd_after.confidence_interval.high:.2f}) \\\\\n"
            )

            alignment_scores_fwd_before = []
            alignment_scores_fwd_after = []
            alignment_scores_bwd_before = []
            alignment_scores_bwd_after = []
            delta_scores = []

            for model, n_layer in model_to_nlayer.items():
                if model not in df["model"].unique():
                    continue
                layer = n_layer - 2
                subdf = df[((df.model == model) & (df.task == task))]

                for lang in langs:
                    alignment_scores_fwd_before += list(
                        subdf[f"alignment_before_fwd_{lang}_{layer}"].dropna()
                    )
                    alignment_scores_fwd_after += list(
                        subdf[f"alignment_after_fwd_{lang}_{layer}"].dropna()
                    )
                    alignment_scores_bwd_before += list(
                        subdf[f"alignment_before_bwd_{lang}_{layer}"].dropna()
                    )
                    alignment_scores_bwd_after += list(
                        subdf[f"alignment_after_bwd_{lang}_{layer}"].dropna()
                    )
                    delta_scores += list(
                        (
                            subdf[f"final_eval_{lang}_{task_to_metric[task]}"]
                            - subdf[f"final_eval_same_{task_to_metric[task]}"]
                        )
                        / subdf[f"final_eval_same_{task_to_metric[task]}"]
                    )

            rho_fwd_before, p_fwd_before = spearmanr(alignment_scores_fwd_before, delta_scores)
            rho_fwd_after, p_fwd_after = spearmanr(alignment_scores_fwd_after, delta_scores)
            rho_bwd_before, p_bwd_before = spearmanr(alignment_scores_bwd_before, delta_scores)
            rho_bwd_after, p_bwd_after = spearmanr(alignment_scores_bwd_after, delta_scores)

            res += (
                " & penult. & "
                + ("\cellcolor{gray!30}" if p_fwd_before > 0.05 else "")
                + f"{rho_fwd_before:.2f} & "
                + ("\cellcolor{gray!30}" if p_fwd_after > 0.05 else "")
                + f"{rho_fwd_after:.2f} & "
                + ("\cellcolor{gray!30}" if p_bwd_before > 0.05 else "")
                + f"{rho_bwd_before:.2f} & "
                + ("\cellcolor{gray!30}" if p_bwd_after > 0.05 else "")
                + f"{rho_bwd_after:.2f} \\\\\n"
            )
            res += "\\hline\n"

            boostrap_fwd_before = spearman_with_bootstrap_ci(
                alignment_scores_fwd_before, delta_scores, n_resamples=2_000
            )
            boostrap_fwd_after = spearman_with_bootstrap_ci(
                alignment_scores_fwd_after, delta_scores, n_resamples=2_000
            )
            boostrap_bwd_before = spearman_with_bootstrap_ci(
                alignment_scores_bwd_before, delta_scores, n_resamples=2_000
            )
            boostrap_bwd_after = spearman_with_bootstrap_ci(
                alignment_scores_bwd_after, delta_scores, n_resamples=2_000
            )

            ci_res += (
                " & penult. "
                + f"& {rho_fwd_before:.2f} ({boostrap_fwd_before.confidence_interval.low:.2f} - "
                + f"{boostrap_fwd_before.confidence_interval.high:.2f}) "
                + f"& {rho_fwd_after:.2f} ({boostrap_fwd_after.confidence_interval.low:.2f} - "
                + f"{boostrap_fwd_after.confidence_interval.high:.2f}) "
                + f"& {rho_bwd_before:.2f} ({boostrap_bwd_before.confidence_interval.low:.2f} - "
                + f"{boostrap_bwd_before.confidence_interval.high:.2f}) "
                + f"& {rho_bwd_after:.2f} ({boostrap_bwd_after.confidence_interval.low:.2f} - "
                + f"{boostrap_bwd_after.confidence_interval.high:.2f}) \\\\\n"
            )

            # For best layer

            alignment_scores_fwd_before = []
            alignment_scores_fwd_after = []
            alignment_scores_bwd_before = []
            alignment_scores_bwd_after = []
            delta_scores = []

            for model, n_layer in model_to_nlayer.items():
                if model not in df["model"].unique():
                    continue
                subdf = df[((df.model == model) & (df.task == task))]

                for lang in langs:
                    alignment_scores_fwd_before += list(
                        np.max(
                            [
                                list(subdf[f"alignment_before_fwd_{lang}_{layer}"].dropna())
                                for layer in range(n_layer)
                            ],
                            axis=0,
                        )
                    )
                    alignment_scores_fwd_after += list(
                        np.max(
                            [
                                list(subdf[f"alignment_after_fwd_{lang}_{layer}"].dropna())
                                for layer in range(n_layer)
                            ],
                            axis=0,
                        )
                    )
                    alignment_scores_bwd_before += list(
                        np.max(
                            [
                                list(subdf[f"alignment_before_bwd_{lang}_{layer}"].dropna())
                                for layer in range(n_layer)
                            ],
                            axis=0,
                        )
                    )
                    alignment_scores_bwd_after += list(
                        np.max(
                            [
                                list(subdf[f"alignment_after_bwd_{lang}_{layer}"].dropna())
                                for layer in range(n_layer)
                            ],
                            axis=0,
                        )
                    )

                    delta_scores += list(
                        (
                            subdf[f"final_eval_{lang}_{task_to_metric[task]}"]
                            - subdf[f"final_eval_same_{task_to_metric[task]}"]
                        )
                        / subdf[f"final_eval_same_{task_to_metric[task]}"]
                    )

            rho_fwd_before, p_fwd_before = spearmanr(alignment_scores_fwd_before, delta_scores)
            rho_fwd_after, p_fwd_after = spearmanr(alignment_scores_fwd_after, delta_scores)
            rho_bwd_before, p_bwd_before = spearmanr(alignment_scores_bwd_before, delta_scores)
            rho_bwd_after, p_bwd_after = spearmanr(alignment_scores_bwd_after, delta_scores)

            res += (
                " & best & "
                + ("\cellcolor{gray!30}" if p_fwd_before > 0.05 else "")
                + f"{rho_fwd_before:.2f} & "
                + ("\cellcolor{gray!30}" if p_fwd_after > 0.05 else "")
                + f"{rho_fwd_after:.2f} & "
                + ("\cellcolor{gray!30}" if p_bwd_before > 0.05 else "")
                + f"{rho_bwd_before:.2f} & "
                + ("\cellcolor{gray!30}" if p_bwd_after > 0.05 else "")
                + f"{rho_bwd_after:.2f} \\\\\n"
            )
            res += "\\hline\n"

            boostrap_fwd_before = spearman_with_bootstrap_ci(
                alignment_scores_fwd_before, delta_scores, n_resamples=2_000
            )
            boostrap_fwd_after = spearman_with_bootstrap_ci(
                alignment_scores_fwd_after, delta_scores, n_resamples=2_000
            )
            boostrap_bwd_before = spearman_with_bootstrap_ci(
                alignment_scores_bwd_before, delta_scores, n_resamples=2_000
            )
            boostrap_bwd_after = spearman_with_bootstrap_ci(
                alignment_scores_bwd_after, delta_scores, n_resamples=2_000
            )

            ci_res += (
                " & best "
                + f"& {rho_fwd_before:.2f} ({boostrap_fwd_before.confidence_interval.low:.2f} - "
                + f"{boostrap_fwd_before.confidence_interval.high:.2f}) "
                + f"& {rho_fwd_after:.2f} ({boostrap_fwd_after.confidence_interval.low:.2f} - "
                + f"{boostrap_fwd_after.confidence_interval.high:.2f}) "
                + f"& {rho_bwd_before:.2f} ({boostrap_bwd_before.confidence_interval.low:.2f} - "
                + f"{boostrap_bwd_before.confidence_interval.high:.2f}) "
                + f"& {rho_bwd_after:.2f} ({boostrap_bwd_after.confidence_interval.low:.2f} - "
                + f"{boostrap_bwd_after.confidence_interval.high:.2f}) \\\\\n"
            )
        print(res)
        print("\n\n")
        print(ci_res)

    if "corr_table_by_seed" in args.plots:
        # Table by task and by moment

        for seed in df["seed"].unique():
            seed_df = df[(df.seed == seed)]

            ci_res = "task & layer & \multicolumn{2}{|c}{en-X} & \multicolumn{2}{|c}{X-en}\\\\\n"
            ci_res += "& & before & after & before & after \\\\\n\\hline\n"

            models_for_layers = ["bert-base-multilingual-cased", "xlm-roberta-base"]

            for task in tasks:
                task_name = task_to_name[task]
                ci_res += f"\\multirow{{2}}{{*}}{{{task_name}}}"

                alignment_scores_fwd_before = []
                alignment_scores_fwd_after = []
                alignment_scores_bwd_before = []
                alignment_scores_bwd_after = []
                delta_scores = []

                for model, n_layer in model_to_nlayer.items():
                    if model not in seed_df["model"].unique():
                        continue

                    layer = n_layer - 1
                    subdf = seed_df[((seed_df.model == model) & (seed_df.task == task))]

                    for lang in langs:
                        alignment_scores_fwd_before += list(
                            subdf[f"alignment_before_fwd_{lang}_{layer}"].dropna()
                        )
                        alignment_scores_fwd_after += list(
                            subdf[f"alignment_after_fwd_{lang}_{layer}"].dropna()
                        )
                        alignment_scores_bwd_before += list(
                            subdf[f"alignment_before_bwd_{lang}_{layer}"].dropna()
                        )
                        alignment_scores_bwd_after += list(
                            subdf[f"alignment_after_bwd_{lang}_{layer}"].dropna()
                        )
                        delta_scores += list(
                            (
                                subdf[f"final_eval_{lang}_{task_to_metric[task]}"]
                                - subdf[f"final_eval_same_{task_to_metric[task]}"]
                            )
                            / subdf[f"final_eval_same_{task_to_metric[task]}"]
                        )

                rho_fwd_before, p_fwd_before = spearmanr(alignment_scores_fwd_before, delta_scores)
                rho_fwd_after, p_fwd_after = spearmanr(alignment_scores_fwd_after, delta_scores)
                rho_bwd_before, p_bwd_before = spearmanr(alignment_scores_bwd_before, delta_scores)
                rho_bwd_after, p_bwd_after = spearmanr(alignment_scores_bwd_after, delta_scores)

                boostrap_fwd_before = spearman_with_bootstrap_ci(
                    alignment_scores_fwd_before, delta_scores, n_resamples=2_000
                )
                boostrap_fwd_after = spearman_with_bootstrap_ci(
                    alignment_scores_fwd_after, delta_scores, n_resamples=2_000
                )
                boostrap_bwd_before = spearman_with_bootstrap_ci(
                    alignment_scores_bwd_before, delta_scores, n_resamples=2_000
                )
                boostrap_bwd_after = spearman_with_bootstrap_ci(
                    alignment_scores_bwd_after, delta_scores, n_resamples=2_000
                )

                ci_res += (
                    " & penult. "
                    + f"& {rho_fwd_before:.2f} ({boostrap_fwd_before.confidence_interval.low:.2f} - "
                    + f"{boostrap_fwd_before.confidence_interval.high:.2f}) "
                    + f"& {rho_fwd_after:.2f} ({boostrap_fwd_after.confidence_interval.low:.2f} - "
                    + f"{boostrap_fwd_after.confidence_interval.high:.2f}) "
                    + f"& {rho_bwd_before:.2f} ({boostrap_bwd_before.confidence_interval.low:.2f} - "
                    + f"{boostrap_bwd_before.confidence_interval.high:.2f}) "
                    + f"& {rho_bwd_after:.2f} ({boostrap_bwd_after.confidence_interval.low:.2f} - "
                    + f"{boostrap_bwd_after.confidence_interval.high:.2f}) \\\\\n"
                )

                alignment_scores_fwd_before = []
                alignment_scores_fwd_after = []
                alignment_scores_bwd_before = []
                alignment_scores_bwd_after = []
                delta_scores = []

                for model, n_layer in model_to_nlayer.items():
                    if model not in seed_df["model"].unique():
                        continue
                    layer = n_layer - 2
                    subdf = seed_df[((seed_df.model == model) & (seed_df.task == task))]

                    for lang in langs:
                        alignment_scores_fwd_before += list(
                            subdf[f"alignment_before_fwd_{lang}_{layer}"].dropna()
                        )
                        alignment_scores_fwd_after += list(
                            subdf[f"alignment_after_fwd_{lang}_{layer}"].dropna()
                        )
                        alignment_scores_bwd_before += list(
                            subdf[f"alignment_before_bwd_{lang}_{layer}"].dropna()
                        )
                        alignment_scores_bwd_after += list(
                            subdf[f"alignment_after_bwd_{lang}_{layer}"].dropna()
                        )
                        delta_scores += list(
                            (
                                subdf[f"final_eval_{lang}_{task_to_metric[task]}"]
                                - subdf[f"final_eval_same_{task_to_metric[task]}"]
                            )
                            / subdf[f"final_eval_same_{task_to_metric[task]}"]
                        )

                rho_fwd_before, p_fwd_before = spearmanr(alignment_scores_fwd_before, delta_scores)
                rho_fwd_after, p_fwd_after = spearmanr(alignment_scores_fwd_after, delta_scores)
                rho_bwd_before, p_bwd_before = spearmanr(alignment_scores_bwd_before, delta_scores)
                rho_bwd_after, p_bwd_after = spearmanr(alignment_scores_bwd_after, delta_scores)

                boostrap_fwd_before = spearman_with_bootstrap_ci(
                    alignment_scores_fwd_before, delta_scores, n_resamples=2_000
                )
                boostrap_fwd_after = spearman_with_bootstrap_ci(
                    alignment_scores_fwd_after, delta_scores, n_resamples=2_000
                )
                boostrap_bwd_before = spearman_with_bootstrap_ci(
                    alignment_scores_bwd_before, delta_scores, n_resamples=2_000
                )
                boostrap_bwd_after = spearman_with_bootstrap_ci(
                    alignment_scores_bwd_after, delta_scores, n_resamples=2_000
                )

                ci_res += (
                    " & penult. "
                    + f"& {rho_fwd_before:.2f} ({boostrap_fwd_before.confidence_interval.low:.2f} - "
                    + f"{boostrap_fwd_before.confidence_interval.high:.2f}) "
                    + f"& {rho_fwd_after:.2f} ({boostrap_fwd_after.confidence_interval.low:.2f} - "
                    + f"{boostrap_fwd_after.confidence_interval.high:.2f}) "
                    + f"& {rho_bwd_before:.2f} ({boostrap_bwd_before.confidence_interval.low:.2f} - "
                    + f"{boostrap_bwd_before.confidence_interval.high:.2f}) "
                    + f"& {rho_bwd_after:.2f} ({boostrap_bwd_after.confidence_interval.low:.2f} - "
                    + f"{boostrap_bwd_after.confidence_interval.high:.2f}) \\\\\n"
                )

                # For best layer

                alignment_scores_fwd_before = []
                alignment_scores_fwd_after = []
                alignment_scores_bwd_before = []
                alignment_scores_bwd_after = []
                delta_scores = []

                for model, n_layer in model_to_nlayer.items():
                    if model not in seed_df["model"].unique():
                        continue
                    subdf = seed_df[((seed_df.model == model) & (seed_df.task == task))]

                    for lang in langs:
                        alignment_scores_fwd_before += list(
                            np.max(
                                [
                                    list(subdf[f"alignment_before_fwd_{lang}_{layer}"].dropna())
                                    for layer in range(n_layer)
                                ],
                                axis=0,
                            )
                        )
                        alignment_scores_fwd_after += list(
                            np.max(
                                [
                                    list(subdf[f"alignment_after_fwd_{lang}_{layer}"].dropna())
                                    for layer in range(n_layer)
                                ],
                                axis=0,
                            )
                        )
                        alignment_scores_bwd_before += list(
                            np.max(
                                [
                                    list(subdf[f"alignment_before_bwd_{lang}_{layer}"].dropna())
                                    for layer in range(n_layer)
                                ],
                                axis=0,
                            )
                        )
                        alignment_scores_bwd_after += list(
                            np.max(
                                [
                                    list(subdf[f"alignment_after_bwd_{lang}_{layer}"].dropna())
                                    for layer in range(n_layer)
                                ],
                                axis=0,
                            )
                        )

                        delta_scores += list(
                            (
                                subdf[f"final_eval_{lang}_{task_to_metric[task]}"]
                                - subdf[f"final_eval_same_{task_to_metric[task]}"]
                            )
                            / subdf[f"final_eval_same_{task_to_metric[task]}"]
                        )

                rho_fwd_before, p_fwd_before = spearmanr(alignment_scores_fwd_before, delta_scores)
                rho_fwd_after, p_fwd_after = spearmanr(alignment_scores_fwd_after, delta_scores)
                rho_bwd_before, p_bwd_before = spearmanr(alignment_scores_bwd_before, delta_scores)
                rho_bwd_after, p_bwd_after = spearmanr(alignment_scores_bwd_after, delta_scores)

                boostrap_fwd_before = spearman_with_bootstrap_ci(
                    alignment_scores_fwd_before, delta_scores, n_resamples=2_000
                )
                boostrap_fwd_after = spearman_with_bootstrap_ci(
                    alignment_scores_fwd_after, delta_scores, n_resamples=2_000
                )
                boostrap_bwd_before = spearman_with_bootstrap_ci(
                    alignment_scores_bwd_before, delta_scores, n_resamples=2_000
                )
                boostrap_bwd_after = spearman_with_bootstrap_ci(
                    alignment_scores_bwd_after, delta_scores, n_resamples=2_000
                )

                ci_res += (
                    " & best "
                    + f"& {rho_fwd_before:.2f} ({boostrap_fwd_before.confidence_interval.low:.2f} - "
                    + f"{boostrap_fwd_before.confidence_interval.high:.2f}) "
                    + f"& {rho_fwd_after:.2f} ({boostrap_fwd_after.confidence_interval.low:.2f} - "
                    + f"{boostrap_fwd_after.confidence_interval.high:.2f}) "
                    + f"& {rho_bwd_before:.2f} ({boostrap_bwd_before.confidence_interval.low:.2f} - "
                    + f"{boostrap_bwd_before.confidence_interval.high:.2f}) "
                    + f"& {rho_bwd_after:.2f} ({boostrap_bwd_after.confidence_interval.low:.2f} - "
                    + f"{boostrap_bwd_after.confidence_interval.high:.2f}) \\\\\n"
                )
            print("\n\n")
            print(ci_res)

    if "best_layers_for_correlation" in args.plots:
        for task, direction, moment, distil_layer in itertools.product(
            tasks, ["fwd"], ["before", "after"], range(7)
        ):
            combinations = []
            correlations = []
            xs = []
            ys = []

            for base_layer, large_layer in itertools.product(range(13), range(25)):
                combinations.append((distil_layer, base_layer, large_layer))

                x = []
                y = []

                for model, n_layer in model_to_nlayer.items():
                    if n_layer == 7:
                        layer = distil_layer
                    elif n_layer == 13:
                        layer = base_layer
                    elif n_layer == 25:
                        layer = large_layer
                    else:
                        print(f"{model} do not have a compatible number of layer: got {n_layer}")
                        continue

                    subdf = df[((df.model == model) & (df.task == task))]

                    for lang in langs:
                        x += list(subdf[f"alignment_{moment}_{direction}_{lang}_{layer}"].dropna())
                        y += list(
                            (
                                subdf[f"final_eval_{lang}_{task_to_metric[task]}"]
                                - subdf[f"final_eval_same_{task_to_metric[task]}"]
                            )
                            / subdf[f"final_eval_same_{task_to_metric[task]}"]
                        )

                correlations.append(spearmanr(x, y)[0])
                xs.append(x)
                ys.append(y)

            max_idx = np.argmax(correlations)

            bounds = spearman_with_bootstrap_ci(
                xs[max_idx], ys[max_idx], n_resamples=2_000
            ).confidence_interval

            print(
                f"{task}, {direction}, {moment}, distil_layer={distil_layer}: {correlations[max_idx]:.2f} ({bounds.low:.2f} - {bounds.high:.2f}) {combinations[max_idx]}"
            )

            low_bound = bounds.low

            similar_ids = []
            for i in range(len(combinations)):
                if i == max_idx:
                    continue
                if correlations[i] > low_bound:
                    similar_ids.append(i)

            print(f"  {len(similar_ids)} overlaps: {[combinations[i] for i in similar_ids[:5]]}")

    if "correlation_by_layer_table" in args.plots:
        # Table by layer and task

        layer_table_dir = os.path.join(args.output_dir, "correlation_by_layer_table")
        Path(layer_table_dir).mkdir(exist_ok=True)

        for task in tasks:
            for direction in ["fwd", "bwd"]:
                res = f"{task_to_name[task]} {direction} & "
                column_heads = map(lambda x: f"\\multicolumn{{2}}{{c|}}{{{x}}}", models.values())
                res += " & ".join(column_heads)
                res += "\\\\\n"
                res += (
                    " & "
                    + " & ".join(map(lambda _: "before & after", models))
                    + " & before & after\\\\\n\\hline\n"
                )

                for to_remove in range(max(model_to_nlayer.values())):
                    res += "last & " if to_remove == "0" else f"-{to_remove} & "

                    for model, n_layers in model_to_nlayer.items():
                        layer = n_layers - to_remove - 1
                        if layer < 0:
                            res += "- & - & "
                            continue

                        subdf = df[(df.model == model) & (df.task == task)]

                        alignment_scores_before = []
                        alignment_scores_after = []
                        delta_scores = []

                        for lang in langs:
                            alignment_scores_before += list(
                                subdf[f"alignment_before_{direction}_{lang}_{layer}"]
                            )
                            alignment_scores_after += list(
                                subdf[f"alignment_after_{direction}_{lang}_{layer}"]
                            )
                            delta_scores += list(
                                (
                                    subdf[f"final_eval_{lang}_{task_to_metric[task]}"]
                                    - subdf[f"final_eval_same_{task_to_metric[task]}"]
                                )
                                / subdf[f"final_eval_same_{task_to_metric[task]}"]
                            )

                        corr_before, p_before = spearmanr(alignment_scores_before, delta_scores)
                        corr_after, p_after = spearmanr(alignment_scores_after, delta_scores)

                        res += (
                            ("\cellcolor{gray!30}" if p_before > 0.05 else "")
                            + f"{corr_before:.2f} & "
                            + ("\cellcolor{gray!30}" if p_after > 0.05 else "")
                            + f" {corr_after:.2f} & "
                        )

                    res += "\\\\\n"

                with open(os.path.join(layer_table_dir, f"{task}_{direction}.txt"), "w") as f:
                    f.write(res)

    for model in models:
        if f"correlation_by_layer_table_{model}" in args.plots:
            layer_table_dir = os.path.join(args.output_dir, f"correlation_by_layer_table_{model}")
            Path(layer_table_dir).mkdir(exist_ok=True)

            for task in tasks:
                res = f"{task_to_name[task]} & "
                res += " & ".join(
                    map(lambda x: f"\\multicolumn{{2}}{{c|}}{{{x}}}", ["forward", "backward"])
                )
                res += "\\\\\n"
                res += " & before & after & before & after \\\\\n"

                for to_remove in range(model_to_nlayer[model]):
                    res += "last " if to_remove == "0" else f"-{to_remove} "

                    layer = model_to_nlayer[model] - to_remove - 1
                    if layer < 0:
                        res += "& - & - "
                        continue

                    subdf = df[(df.model == model) & (df.task == task)]

                    for direction in ["fwd", "bwd"]:
                        alignment_scores_before = []
                        alignment_scores_after = []
                        delta_scores = []

                        for lang in langs:
                            alignment_scores_before += list(
                                subdf[f"alignment_before_{direction}_{lang}_{layer}"]
                            )
                            alignment_scores_after += list(
                                subdf[f"alignment_after_{direction}_{lang}_{layer}"]
                            )
                            delta_scores += list(
                                (
                                    subdf[f"final_eval_{lang}_{task_to_metric[task]}"]
                                    - subdf[f"final_eval_same_{task_to_metric[task]}"]
                                )
                                / subdf[f"final_eval_same_{task_to_metric[task]}"]
                            )

                        corr_before, p_before = spearmanr(alignment_scores_before, delta_scores)
                        corr_after, p_after = spearmanr(alignment_scores_after, delta_scores)

                        boostrap_before = spearman_with_bootstrap_ci(
                            alignment_scores_before, delta_scores, n_resamples=2_000
                        )
                        boostrap_after = spearman_with_bootstrap_ci(
                            alignment_scores_after, delta_scores, n_resamples=2_000
                        )

                        res += (
                            f"& {corr_before:.2f} ({boostrap_before.confidence_interval.low:.2f} - "
                            + f"{boostrap_before.confidence_interval.high:.2f}) "
                            + f"& {corr_after:.2f} ({boostrap_after.confidence_interval.low:.2f} - "
                            + f"{boostrap_after.confidence_interval.high:.2f})"
                        )
                    res += "\\\\\n"

                with open(os.path.join(layer_table_dir, f"{task}.txt"), "w") as f:
                    f.write(res)

    if "correlation_charts" in args.plots:
        # Figures of correlation
        corr_chart_dir = os.path.join(args.output_dir, "correlation_charts")
        Path(corr_chart_dir).mkdir(exist_ok=True)

        for task in tasks:
            task_dir = os.path.join(corr_chart_dir, task)
            Path(task_dir).mkdir(exist_ok=True)

            for to_last in range(max(model_to_nlayer.values())):
                available_models = list(
                    filter(lambda x: model_to_nlayer[x] > to_last, models.keys())
                )

                for moment in ["before", "after"]:
                    line_for_models = []
                    line_for_langs = []

                    for j, model in enumerate(available_models):
                        model_name = models[model]
                        subdf = df[(df.model == model) & (df.task == task)]
                        layer = model_to_nlayer[model] - to_last - 1

                        line_for_models.append(
                            mlines.Line2D(
                                [],
                                [],
                                color="black",
                                marker=MarkerStyle.filled_markers[j],
                                linestyle="",
                            )
                        )

                        for i, lang in enumerate(langs):
                            alignment_scores = list(subdf[f"alignment_{moment}_fwd_{lang}_{layer}"])
                            delta_scores = list(
                                (
                                    subdf[f"final_eval_{lang}_{task_to_metric[task]}"]
                                    - subdf[f"final_eval_same_{task_to_metric[task]}"]
                                )
                                / subdf[f"final_eval_same_{task_to_metric[task]}"]
                            )
                            (line,) = plt.plot(
                                alignment_scores,
                                delta_scores,
                                color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
                                marker=MarkerStyle.filled_markers[j],
                                linestyle=""
                                # label=f"{model_name} {lang}",
                            )
                            if j == 0:
                                line_for_langs.append(line)

                    legend1 = plt.legend(
                        line_for_langs,
                        langs,
                        loc="lower right",
                        bbox_to_anchor=(0.80, 0.05, 0.15, 0.30),
                    )
                    plt.legend(
                        line_for_models,
                        list(map(models.__getitem__, available_models)),
                        loc="lower right",
                        bbox_to_anchor=(0.50, 0.05, 0.25, 0.30),
                    )
                    plt.gca().add_artist(legend1)
                    plt.xlabel("alignment score")
                    plt.ylabel("cross-lingual generalization")
                    plt.savefig(os.path.join(task_dir, f"{to_last}_tolast_{moment}_fwd.pdf"))
                    plt.clf()
                    line_for_models = []
                    line_for_langs = []

                    for j, model in enumerate(available_models):
                        model_name = models[model]
                        subdf = df[(df.model == model) & (df.task == task)]
                        layer = model_to_nlayer[model] - to_last - 1

                        line_for_models.append(
                            mlines.Line2D(
                                [],
                                [],
                                color="black",
                                marker=MarkerStyle.filled_markers[j],
                                linestyle="",
                            )
                        )

                        for i, lang in enumerate(langs):
                            alignment_scores = list(subdf[f"alignment_{moment}_bwd_{lang}_{layer}"])
                            delta_scores = list(
                                (
                                    subdf[f"final_eval_{lang}_{task_to_metric[task]}"]
                                    - subdf[f"final_eval_same_{task_to_metric[task]}"]
                                )
                                / subdf[f"final_eval_same_{task_to_metric[task]}"]
                            )
                            (line,) = plt.plot(
                                alignment_scores,
                                delta_scores,
                                color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
                                marker=MarkerStyle.filled_markers[j],
                                linestyle=""
                                # label=f"{model_name} {lang}",
                            )
                            if j == 0:
                                line_for_langs.append(line)

                    legend1 = plt.legend(
                        line_for_langs,
                        langs,
                        loc="lower right",
                        bbox_to_anchor=(0.80, 0.05, 0.15, 0.30),
                    )
                    plt.legend(
                        line_for_models,
                        list(map(models.__getitem__, available_models)),
                        loc="lower right",
                        bbox_to_anchor=(0.50, 0.05, 0.25, 0.30),
                    )
                    plt.gca().add_artist(legend1)
                    plt.xlabel("alignment score")
                    plt.ylabel("cross-lingual generalization")
                    plt.savefig(os.path.join(task_dir, f"{to_last}_tolast_{moment}_bwd.pdf"))
                    plt.clf()

    if "before_vs_after" in args.plots:
        # Best alignment before vs. best alignment after

        before_vs_after_dir = os.path.join(args.output_dir, "before_vs_after")
        Path(before_vs_after_dir).mkdir(exist_ok=True)

        for task in tasks:
            for direction in ["fwd", "bwd"]:
                all_before = []
                all_after = []

                for j, (model, model_name) in enumerate(models.items()):
                    n_layers = model_to_nlayer[model]

                    subdf = df[((df.model == model) & (df.task == task))]

                    for i, lang in enumerate(langs):
                        values_before = np.array(
                            [
                                list(subdf[f"alignment_before_{direction}_{lang}_{i}"].dropna())
                                for i in range(n_layers)
                            ]
                        )
                        values_after = np.array(
                            [
                                list(subdf[f"alignment_after_{direction}_{lang}_{i}"].dropna())
                                for i in range(n_layers)
                            ]
                        )

                        best_before = np.max(values_before, axis=0)
                        best_after = np.max(values_after, axis=0)

                        all_before += list(best_before)
                        all_after += list(best_after)

                        plt.scatter(
                            best_before,
                            best_after,
                            color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
                            marker=MarkerStyle.filled_markers[j],
                            label=f"{model_name} {lang}",
                        )
                plt.plot([0, 1], [0, 1], color="black")
                plt.xlim([0, 1.0])
                plt.ylim([0, 1.0])
                plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0))
                plt.tight_layout()
                plt.xlabel("alignment score before")
                plt.ylabel("alignment score after")
                plt.savefig(os.path.join(before_vs_after_dir, f"{task}_{direction}.pdf"))
                plt.clf()

                rho, p = spearmanr(all_before, all_after)
                bootstrap_result = spearman_with_bootstrap_ci(
                    all_before, all_after, n_resamples=2_000
                )

                print(
                    f"best before vs after {task} {direction}: rho={rho:.2f}, p={p:.3f}, ci=({bootstrap_result.confidence_interval.low:.2f} - {bootstrap_result.confidence_interval.high:.2f})"
                )

    if "before_vs_after_last" in args.plots:
        before_vs_after_dir = os.path.join(args.output_dir, "before_vs_after_last")
        Path(before_vs_after_dir).mkdir(exist_ok=True)

        for task in tasks:
            for direction in ["fwd", "bwd"]:
                all_before = []
                all_after = []

                for j, (model, model_name) in enumerate(models.items()):
                    n_layers = model_to_nlayer[model]

                    subdf = df[((df.model == model) & (df.task == task))]

                    for i, lang in enumerate(langs):
                        values_before = np.array(
                            subdf[f"alignment_before_{direction}_{lang}_{n_layers - 1}"].dropna()
                        )
                        values_after = np.array(
                            subdf[f"alignment_after_{direction}_{lang}_{n_layers - 1}"].dropna()
                        )

                        all_before += list(values_before)
                        all_after += list(values_after)

                        plt.scatter(
                            values_before,
                            values_after,
                            color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
                            marker=MarkerStyle.filled_markers[j],
                            label=f"{model_name} {lang}",
                        )
                plt.plot([0, 1], [0, 1], color="black")
                plt.xlim([0, 1.0])
                plt.ylim([0, 1.0])
                plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0))
                plt.tight_layout()
                plt.xlabel("alignment score before")
                plt.ylabel("alignment score after")
                plt.savefig(os.path.join(before_vs_after_dir, f"{task}_{direction}.pdf"))
                plt.clf()

                rho, p = spearmanr(all_before, all_after)
                bootstrap_result = spearman_with_bootstrap_ci(
                    all_before, all_after, n_resamples=2_000
                )

                print(
                    f"last before vs after {task} {direction}: rho={rho:.2f}, p={p:.3f}, ci=({bootstrap_result.confidence_interval.low:.2f} - {bootstrap_result.confidence_interval.high:.2f})"
                )
