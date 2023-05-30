import os
import sys

import numpy as np
import pandas as pd
from transformers import AutoModel

sys.path.append(os.curdir)

from multilingual_eval.utils import get_nb_layers

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("csv_files", nargs="+")
    parser.add_argument("--langs", type=str, nargs="+", default=["ar", "es", "fr", "ru", "zh"])
    parser.add_argument("--tasks", type=str, nargs="+", default=["udpos", "wikiann", "xnli"])
    args = parser.parse_args()

    df = pd.concat([pd.read_csv(fname) for fname in args.csv_files])

    task_to_name = {"wikiann": "NER", "udpos": "POS", "xnli": "NLI"}

    model_to_nlayer = {}

    models = {
        "distilbert-base-multilingual-cased": "distilmBERT",
        "bert-base-multilingual-cased": "mBERT",
        "xlm-roberta-base": "XLM-R Base",
        "xlm-roberta-large": "XLM-R Large",
    }

    model_in_df = set(df["model"].unique())

    models = {key: value for key, value in models.items() if key in model_in_df}

    for model_name in models:
        model = AutoModel.from_pretrained(model_name, cache_dir=args.cache_dir)
        n_layers = get_nb_layers(model)
        model_to_nlayer[model_name] = n_layers

    res = "task & model & "

    res += " & ".join(map(lambda x: f"en-{x}", args.langs))
    res += "\\\\\n"
    res += "\\hline"

    for task in args.tasks:
        task_name = task_to_name[task]
        res += f"\\multirow{{{len(models)}}}{{*}}{{{task_name}}}"
        for model, model_name in models.items():
            if model not in df["model"].unique():
                continue
            last_layer = model_to_nlayer[model] - 1
            res += f"& {model_name}"
            subdf = df[(df.model == model) & (df.task == task)]
            for lang in args.langs:
                res += " & "
                lang_drop = (
                    subdf[f"alignment_after_fwd_{lang}_{last_layer}"].dropna()
                    - subdf[f"alignment_before_fwd_{lang}_{last_layer}"].dropna()
                ) / subdf[f"alignment_before_fwd_{lang}_{last_layer}"].dropna()
                res += f" {np.mean(lang_drop):.2f}$_{{\pm {np.std(lang_drop):.2f}}}$"
            res += "\\\\\n"
        res += "\\hline"

    print(res)