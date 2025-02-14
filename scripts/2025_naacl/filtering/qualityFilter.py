import os, shutil
import argparse
from pathlib import Path
from comet import download_model, load_from_checkpoint
import itertools
from contextlib import ExitStack
import torch
import numpy as np

def prepare_directories(data_dir: str, dataset: str, subdirs, percentiles):
    for subdir, perc in itertools.product(subdirs, percentiles):
        new_dir = os.path.join(data_dir, subdir, f"{dataset}_filtered_percent_{perc}")
        Path(new_dir).mkdir(exist_ok=True, parents=True)


def get_languages(langs, translation_dir, left_lang):
    if langs:
        return langs
    return list(
        map(
            lambda x: x.split(".")[0].split("-")[1],
            filter(
                lambda x: x.startswith(f"{left_lang}-")
                and x.endswith(".tokenized.train.txt"),
                os.listdir(translation_dir),
            ),
        )
    )


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')

    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    model = load_from_checkpoint(model_path)

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("dataset", type=str)
    parser.add_argument("--translation_dir", type=str, default="translation")
    parser.add_argument(
        "--alignment_dirs",
        type=str,
        default=["awesome-align", "dico-align", "fastalign"],
    )
    parser.add_argument("--left_lang", type=str, default="en")
    parser.add_argument("--right_langs", type=str, nargs="+", default=None)
    parser.add_argument(
        "--percentiles", type=float, nargs="+", default=[25, 37, 50, 62, 75]
    )
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    prepare_directories(
        args.data_dir,
        args.dataset,
        [args.translation_dir, *args.alignment_dirs],
        args.percentiles,
    )

    translation_dir = os.path.join(args.data_dir, args.translation_dir, args.dataset)
    alignment_dirs = [os.path.join(args.data_dir, subdir, args.dataset) for subdir in args.alignment_dirs]

    languages = get_languages(args.right_langs, translation_dir, args.left_lang)

    for lang in languages:
        with open(
            os.path.join(
                translation_dir, f"{args.left_lang}-{lang}.tokenized.train.txt"
            ),
            "r",
        ) as file:
            # Loop through each line in the file
            data = []
            for line in file:
                data.append(
                    {
                        "src": line.strip().split("|||")[0],
                        "mt": line.strip().split("|||")[1],
                    }
                )
        alignment_data = []
        for alignment_dir in alignment_dirs:
            this_data = []
            with open(
                os.path.join(
                    alignment_dir,
                    f"{args.left_lang}-{lang}.train",
                )
            ) as file:
                for line in file:
                    this_data.append(line.strip())
            alignment_data.append(this_data)

        model_output = model.predict(data, batch_size=args.batch_size, gpus=1, num_workers=os.cpu_count() - 1)

        scores = model_output[0]

        thresholds = np.percentile(scores, args.percentiles)

        line_counters = [0] * len(thresholds)
        with ExitStack() as stack:
            translation_writers = [
                stack.enter_context(
                    open(
                        os.path.join(
                            args.data_dir,
                            args.translation_dir,
                            f"{args.dataset}_filtered_percent_{p}",
                            f"{args.left_lang}-{lang}.tokenized.train.txt",
                        ),
                        "w",
                    )
                )
                for p in args.percentiles
            ]
            alignment_writers = [
                [
                    stack.enter_context(
                        open(
                            os.path.join(
                                args.data_dir,
                                subdir,
                                f"{args.dataset}_filtered_percent_{p}",
                                f"{args.left_lang}-{lang}.train",
                            ),
                            "w",
                        )
                    )
                    for subdir in args.alignment_dirs
                ]
                for p in args.percentiles
            ]

            for i_data in range(len(data)):
                line = f"{data[i_data]['src']} ||| {data[i_data]['mt']}\n"
                for i_threshold, t in enumerate(thresholds):
                    if model_output[0][i_data] < t:
                        break
                    line_counters[i_threshold] += 1
                    translation_writers[i_threshold].write(line)
                    for i_alignment, writer in enumerate(
                        alignment_writers[i_threshold]
                    ):
                        writer.write(f"{alignment_data[i_alignment][i_data]}\n")

            for i_t, t in enumerate(thresholds):
                print(
                    f"lang {lang}: filtered out {len(data) - line_counters[i_t]} pairs out of {len(data)}"
                )
