import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import os
import sys
from transformers import AutoModel
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr

sys.path.append(os.curdir)

from multilingual_eval.plotting_utils import COLOR_PALETTE
from multilingual_eval.utils import get_nb_layers


def get_scientific_notation_with_exponent(number, decimals=1):
    res = f"{number:.{decimals}E}"
    left, right = res.split("E")
    return f"{left}\\times 10^{{{right}}}"


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file")
    parser.add_argument("--bar_chart_lang", type=str, default="zh")
    parser.add_argument("--bar_chart_direction", type=str, default="fwd")
    parser.add_argument("--bar_chart_task", type=str, default="udpos")
    parser.add_argument("--bar_chart_model", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--output_dir", type=str, default="local_runs/results")
    parser.add_argument("--moment", type=str, default="after")
    parser.add_argument("--langs", type=str, nargs="+", default=["ar", "es", "fr", "ru", "zh"])
    parser.add_argument("--layer", type=str, default="-1")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)

    # TODO bar chart with before-after for one language, one model
    model = AutoModel.from_pretrained(args.bar_chart_model)
    n_layers = get_nb_layers(model)
    bar_chart_df = df[(df.model == args.bar_chart_model) & (df.task == args.bar_chart_task)]

    property_names_before = []
    property_names_after = []
    layers = []
    for c in bar_chart_df.columns:
        if c.startswith(f"alignment_before_{args.bar_chart_direction}_{args.bar_chart_lang}"):
            property_name_before = c
            property_name_after = f"alignment_after_{args.bar_chart_direction}_{args.bar_chart_lang}_{c.split('_')[-1]}"

            if property_name_after not in bar_chart_df.columns:
                continue

            property_names_before.append(property_name_before)
            property_names_after.append(property_name_after)
            layers.append(n_layers + int(c.split("_")[-1]))

    property_names_after.sort(key=lambda x: n_layers + int(x.split("_")[-1]) + 1)
    property_names_before.sort(key=lambda x: n_layers + int(x.split("_")[-1]) + 1)
    layers.sort()

    mean_after = [np.mean(bar_chart_df[c]) for c in property_names_after]
    mean_before = [np.mean(bar_chart_df[c]) for c in property_names_before]
    std_after = [np.std(bar_chart_df[c]) for c in property_names_after]
    std_before = [np.std(bar_chart_df[c]) for c in property_names_before]

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
    plt.savefig(os.path.join(args.output_dir, "05_a_align_bar_chart.png"))
    plt.clf()

    # TODO table with drop for last layer and all language mBERT and XLM-R
    res = "task & model & "
    langs = args.langs
    if not isinstance(langs, list):
        langs = list(langs)

    task_to_name = {"wikiann": "NER", "udpos": "POS", "xnli": "NLI", "pawsx": "PAWS"}

    res += " & ".join(map(lambda x: f"en-{x}", langs))
    res += "\\\\\n"
    res += "\\hline"

    tasks = df["task"].unique()
    models = {
        "bert-base-multilingual-cased": "mBERT",
        "xlm-roberta-base": "XLM-R Base",
        "xlm-roberta-large": "XLM-R Large",
        # "distilbert-base-multilingual-cased": "distilmBERT",
    }

    for task in tasks:
        task_name = task_to_name[task]
        res += f"\\multirow{{{len(models)}}}{{*}}{{{task_name}}}"
        for model, model_name in models.items():
            if model not in df["model"].unique():
                continue
            res += f"& {model_name}"
            subdf = df[(df.model == model) & (df.task == task)]
            for lang in langs:
                res += " & "
                lang_drop = (
                    subdf[f"alignment_after_fwd_{lang}_-1"]
                    - subdf[f"alignment_before_fwd_{lang}_-1"]
                )
                res += f" {np.mean(lang_drop)*100:.1f}$_{{\pm {np.std(lang_drop)*100:.1f}}}$"
            res += "\\\\\n"
        res += "\\hline"

    print(res)

    print("\n\n")

    # TODO compute correlation between alignment and generalization

    res = "task & model & en-X & X-en\\\\\n"
    res += "\\hline\n"

    task_to_metric = defaultdict(lambda: "accuracy")
    task_to_metric["wikiann"] = "f1"

    best_corr = None
    best_corr_infos = None

    worst_corr = None
    worst_corr_infos = None

    for task in tasks:
        task_name = task_to_name[task]
        res += f"\\multirow{{{len(models)}}}{{*}}{{{task_name}}}"
        for model, model_name in models.items():
            if model not in df["model"].unique():
                continue
            res += f"& {model_name}"
            subdf = df[(df.model == model) & (df.task == task)]
            alignment_scores_bwd = []
            alignment_scores_fwd = []
            en_scores = []
            tgt_scores = []
            for lang in langs:
                alignment_scores_fwd += list(
                    subdf[f"alignment_{args.moment}_fwd_{lang}_{args.layer}"].dropna()
                )
                alignment_scores_bwd += list(
                    subdf[f"alignment_{args.moment}_bwd_{lang}_{args.layer}"].dropna()
                )
                en_scores += list(subdf[f"final_eval_same_{task_to_metric[task]}"].dropna())
                tgt_scores += list(subdf[f"final_eval_{lang}_{task_to_metric[task]}"].dropna())
            delta_scores = list(np.array(tgt_scores) - np.array(en_scores))
            rho_fwd, p_fwd = spearmanr(alignment_scores_fwd, delta_scores)
            rho_bwd, p_bwd = spearmanr(alignment_scores_bwd, delta_scores)

            if best_corr is None or best_corr < rho_fwd:
                best_corr = rho_fwd
                best_corr_infos = (task, model, "fwd_")
            if best_corr is None or best_corr < rho_bwd:
                best_corr = rho_bwd
                best_corr_infos = (task, model, "bwd_")
            if worst_corr is None or worst_corr > rho_fwd:
                worst_corr = rho_fwd
                worst_corr_infos = (task, model, "fwd_")
            if worst_corr is None or worst_corr > rho_bwd:
                worst_corr = rho_bwd
                worst_corr_infos = (task, model, "bwd_")

            res += (
                " & "
                + ("\cellcolor{gray!30}" if p_fwd > 0.05 else "")
                + f"{rho_fwd:.2f} & "
                + ("\cellcolor{gray!30}" if p_bwd > 0.05 else "")
                + f"{rho_bwd:.2f}\\\\\n"
            )
        res += "\\hline\n"

    print(res)

    print("\n\n")

    # Table by task and by layer
    layers = [-1, -3, -5, -7]

    res = "task & layer & en-X & X-en\\\\\n\\hline\n"

    for task in tasks:
        task_name = task_to_name[task]
        subdf = df[df.task == task]
        res += f"\\multirow{{{len(layers)}}}{{*}}{{{task_name}}}"
        for layer in layers:
            res += f"& {layer}"
            alignment_scores_bwd = []
            alignment_scores_fwd = []
            en_scores = []
            tgt_scores = []
            for lang in langs:
                alignment_scores_fwd += list(
                    subdf[f"alignment_{args.moment}_fwd_{lang}_{layer}"].dropna()
                )
                alignment_scores_bwd += list(
                    subdf[f"alignment_{args.moment}_bwd_{lang}_{layer}"].dropna()
                )
                en_scores += list(subdf[f"final_eval_same_{task_to_metric[task]}"].dropna())
                tgt_scores += list(subdf[f"final_eval_{lang}_{task_to_metric[task]}"].dropna())
            delta_scores = list(np.array(tgt_scores) - np.array(en_scores))
            rho_fwd, p_fwd = spearmanr(alignment_scores_fwd, delta_scores)
            rho_bwd, p_bwd = spearmanr(alignment_scores_bwd, delta_scores)

            res += (
                " & "
                + ("\cellcolor{gray!30}" if p_fwd > 0.05 else "")
                + f"{rho_fwd:.2f} & "
                + ("\cellcolor{gray!30}" if p_bwd > 0.05 else "")
                + f"{rho_bwd:.2f}\\\\\n"
            )
        res += "\\hline\n"
    print(res)

    # Figures of correlation
    for task in tasks:
        for j, (model, model_name) in enumerate(models.items()):
            subdf = df[(df.model == model) & (df.task == task)]
            for i, lang in enumerate(langs):

                alignment_scores = list(subdf[f"alignment_{args.moment}_fwd_{lang}_{args.layer}"])
                delta_scores = list(
                    subdf[f"final_eval_{lang}_{task_to_metric[task]}"]
                    - subdf[f"final_eval_same_{task_to_metric[task]}"]
                )
                plt.scatter(
                    alignment_scores,
                    delta_scores,
                    color=COLOR_PALETTE[i],
                    marker=MarkerStyle.filled_markers[j],
                    label=f"{model_name} {lang}",
                )

        plt.legend(loc="lower right")
        plt.xlabel("alignment score")
        plt.ylabel("cross-lingual generalization")
        plt.savefig(os.path.join(args.output_dir, f"05_b_corr_fwd_{task}.png"))
        plt.clf()

        for j, (model, model_name) in enumerate(models.items()):
            subdf = df[(df.model == model) & (df.task == task)]
            for i, lang in enumerate(langs):

                alignment_scores = list(subdf[f"alignment_{args.moment}_bwd_{lang}_{args.layer}"])
                delta_scores = list(
                    subdf[f"final_eval_{lang}_{task_to_metric[task]}"]
                    - subdf[f"final_eval_same_{task_to_metric[task]}"]
                )
                plt.scatter(
                    alignment_scores,
                    delta_scores,
                    color=COLOR_PALETTE[i],
                    marker=MarkerStyle.filled_markers[j],
                    label=f"{model_name} {lang}",
                )

        plt.legend(loc="lower right")
        plt.xlabel("alignment score")
        plt.ylabel("cross-lingual generalization")
        plt.savefig(os.path.join(args.output_dir, f"05_b_corr_bwd_{task}.png"))
        plt.clf()
