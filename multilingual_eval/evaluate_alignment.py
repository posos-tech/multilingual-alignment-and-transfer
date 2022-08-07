import numpy as np

from multilingual_eval.contextualized_pairs import (
    compute_pair_representations,
    generate_pairs,
)
from multilingual_eval.retrieval import evaluate_alignment_with_cosim
from multilingual_eval.utils import get_nb_layers, get_tokenizer_type


def select_pairs_for_evaluation(
    tokenizer,
    translation_dataset,
    dico_path,
    left_lang,
    right_lang,
    nb_selected=5000,
    max_length=512,
):
    def remove_bad_samples(example):
        if example[left_lang] is None:
            return False
        if example[right_lang] is None:
            return False

        left_subwords = tokenizer.tokenize(example[left_lang])
        right_subwords = tokenizer.tokenize(example[right_lang])

        if len(left_subwords) > max_length or len(right_subwords) > max_length:
            return False
        if (
            len(list(filter(lambda x: x != "[UNK]", left_subwords))) == 0
            or len(list(filter(lambda x: x != "[UNK]", right_subwords))) == 0
        ):
            return False
        return True

    translation_dataset = translation_dataset.filter(remove_bad_samples)

    pairs = generate_pairs(
        translation_dataset,
        dico_path,
        left_lang=left_lang,
        right_lang=right_lang,
        avoid_repetition=False,
        max_pairs=nb_selected * 50,
    )

    if len(pairs) < 5000:
        raise Exception(f"Not enough pair with pair {left_lang}-{right_lang}")

    selected_pairs = np.random.choice(pairs, size=(nb_selected,), replace=False)
    return selected_pairs


def evaluate_alignment_on_pairs(
    model,
    tokenizer,
    selected_pairs,
    left_lang,
    right_lang,
    batch_size=2,
    device="cpu:0",
    strong_alignment=False,
    csls_k=10,
):
    left_embs, right_embs = compute_pair_representations(
        model,
        tokenizer,
        selected_pairs,
        batch_size=batch_size,
        dim_size=model.config.hidden_size,
        n_layers=get_nb_layers(model),
        device=device,
        left_lang=left_lang,
        right_lang=right_lang,
        split_type=get_tokenizer_type(tokenizer),
    )

    model = model.cpu()

    res = []
    for layer in range(get_nb_layers(model)):
        score = evaluate_alignment_with_cosim(
            left_embs[layer],
            right_embs[layer],
            device=device,
            csls_k=csls_k,
            strong_alignment=1.0 if strong_alignment else 0.0,
        )
        res.append(score)
    return res
