from typing import Dict, Union, List, Optional

from datasets import get_dataset_infos, load_dataset


def get_news_commentary(
    lang1: str,
    lang2: str,
):

    subsets = set(get_dataset_infos("news_commentary").keys())

    candidate_subset = "-".join(sorted([lang1, lang2]))

    if candidate_subset not in subsets:
        raise Exception(f"pair {candidate_subset} is not available in the `new_commentary` dataset")

    news_commentary = load_dataset("news_commentary", candidate_subset, streaming=True)["train"]

    def preprocess_news_commentary(example):
        return {k: v for k, v in example["translation"].items()}

    news_commentary = (
        news_commentary.map(preprocess_news_commentary)
        .remove_columns(["id", "translation"])
        .filter(lambda x: x[lang1] is not None and x[lang2] is not None)
    )

    return news_commentary
