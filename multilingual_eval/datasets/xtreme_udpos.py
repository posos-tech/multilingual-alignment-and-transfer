from typing import List, Union
import pycountry

from datasets import load_dataset, get_dataset_config_names

from multilingual_eval.datasets.token_classification import get_token_classification_getter


get_xtreme_udpos = get_token_classification_getter(
    lambda lang, cache_dir=None: load_dataset(
        "xtreme",
        f"udpos.{pycountry.languages.get(alpha_2=lang).name}",
        cache_dir=cache_dir,
    ),
    "pos_tags",
)

wuetal_subsets = {
    "en": "en_ewt",
    "ar": "ar_padt",
    "es": "es_gsd",
    "fr": "fr_gsd",
    "ru": "ru_gsd",
    "zh": "zh_gsd",
}

get_wuetal_udpos = get_token_classification_getter(
    lambda lang, cache_dir=None: load_dataset(
        "universal_dependencies", wuetal_subsets[lang], cache_dir=cache_dir
    ),
    "upos",
)


def get_xtreme_udpos_langs():
    """
    Get the list of availabel languages in xtreme.udpos
    """
    return list(
        map(
            lambda x: "el"
            if x.split(".")[1] == "Greek"
            else pycountry.languages.get(name=x.split(".")[1]).alpha_2,
            filter(lambda x: x.startswith("udpos."), get_dataset_config_names("xtreme")),
        )
    )
