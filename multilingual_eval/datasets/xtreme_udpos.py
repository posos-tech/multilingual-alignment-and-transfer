from typing import List, Union
import pycountry

from datasets import load_dataset, interleave_datasets

from multilingual_eval.datasets.label_alignment import LabelAlignmentMapper


def get_xtreme_udpos(
    lang: Union[List[str], str],
    tokenizer,
    limit=None,
    split="train",
    datasets_cache_dir=None,
):

    if not isinstance(lang, list):
        lang = [lang]

    datasets = [
        load_dataset(
            "xtreme",
            f"udpos.{pycountry.languages.get(alpha_2=lang).name}",
            cache_dir=datasets_cache_dir,
        )[split]
        for lang in lang
    ]

    if limit:
        limits = [
            limit // len(datasets) + (1 if i < limit % len(datasets) else 0)
            for i in range(len(datasets))
        ]

        datasets = map(
            lambda x: x[0].filter(lambda _, i: i < x[1] // len(datasets) + 1, with_indices=True),
            zip(datasets, limits),
        )

    datasets = list(
        map(
            lambda x: x.map(LabelAlignmentMapper(tokenizer, label_name="pos_tags"), batched=True),
            datasets,
        ),
    )

    if len(datasets) == 1:
        return datasets[0]
    return interleave_datasets(datasets)
