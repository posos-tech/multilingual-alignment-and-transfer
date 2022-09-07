import torch
import numpy as np
import os
from typing import List, Tuple, Optional, Set
from transformers import AutoTokenizer
from datasets import load_metric
import logging


def get_nb_layers(model):
    res = 1 + model.config.num_hidden_layers
    if hasattr(model.config, "decoder_layers"):
        res += model.config.decoder_layers + 1
    return res


def get_tokenizer_type(tokenizer):
    """
    Get the type of tokenizer (Word-piece, sentence-piece or weird other stuff)
    """
    tokens = tokenizer.tokenize("supercalificious")
    if tokens[0][0] == "▁":
        split_type = "sentencepiece"
    elif len(tokens) > 1 and tokens[1][:2] == "##":
        split_type = "wordpiece"
    elif tokens[-1][-4:] == "</w>":
        split_type = "html_like"
    else:
        raise NotImplementedError(f"Unrecognized tokenizer type: 'supercalificious' -> `{tokens}`")
    return split_type


def find_lang_key_for_mbart_like(tokenizer, lang):
    """
    Finds the id of the special token indicating the language of the sentence
    This is used in models like mBART
    (if applied to a model that do not support this, it will simply return None)
    """
    key = None
    if hasattr(tokenizer, "lang_to_code_id"):
        candidate_keys = list(
            filter(lambda x: x.split("_")[0] == lang, tokenizer.lang_to_code_id.values())
        )
        if len(candidate_keys) != 1:
            raise Exception(
                f"Could not find the right key for language `{lang}` in tokenizer lang_to_code_id: `{list(tokenizer.lang_to_code_id.values())}"
            )
        key = candidate_keys[0]
    return key


def compute_model_hidden_reprs(inputs, model, tokenizer, lang, device="cpu", lang_key=None):
    """
    Computes the hidden representations of a given models for given input representations
    Takes care of various model specificities
    """

    if hasattr(tokenizer, "lang2id"):
        inputs["langs"] = tokenizer.lang2id[lang] * torch.ones(
            inputs["input_ids"].size(), dtype=int
        ).to(device)

    if hasattr(tokenizer, "lang_to_code_id"):
        inputs["input_ids"][:, 0] = tokenizer.lang_to_code_id[lang_key]

    res = model(**inputs, output_hidden_states=True)
    if hasattr(res, "encoder_hidden_states"):
        hidden_repr = res.encoder_hidden_states + res.decoder_hidden_states
    else:
        hidden_repr = res.hidden_states
    return hidden_repr


def subwordlist_to_wordlist(subwordlist: List[str], split_type="wordpiece"):
    """
    Takes a subword list typically output by the tokenize method of a Tokenizer
    and return the list of words and their start and end position in the subword list
    """
    wordlist = []
    word_positions: List[Tuple[int, int]] = []
    current_word = ""
    start_pos = 0
    for i, subword in enumerate(subwordlist):
        if split_type == "wordpiece":
            if subword[:2] == "##":
                current_word += subword[2:]
            elif len(current_word) > 0:
                wordlist.append(current_word)
                word_positions.append((start_pos, i))
                current_word = subword
                start_pos = i
            else:
                current_word = subword
        elif split_type == "sentencepiece":
            if subword[0] == "▁":
                if len(current_word) > 0:
                    wordlist.append(current_word)
                    word_positions.append((start_pos, i + 1))
                current_word = subword[1:]
                start_pos = i
            else:
                current_word += subword
        elif split_type == "html_like":
            if subword[-4:] == "</w>":
                current_word += subword[:-4]
                wordlist.append(current_word)
                word_positions.append((start_pos, i + 1))
                current_word = ""
                start_pos = i
            else:
                current_word += subword
    if len(current_word) > 0:
        wordlist.append(current_word)
        word_positions.append((start_pos, len(subwordlist)))
    return wordlist, word_positions


class UniversalTokenizer:
    def __init__(self, base_tokenizer: str = "bert-base-multilingual-uncased", cache_dir=None):
        self.base_tokenizer = AutoTokenizer.from_pretrained(
            base_tokenizer, **({"cache_dir": cache_dir} if cache_dir is not None else {})
        )
        self.tokenizer_type = get_tokenizer_type(self.base_tokenizer)

    def tokenize(self, sentence, already_tokenized_by_space=False, return_offsets=False):
        """
        Optionally returns character offsets
        """
        if already_tokenized_by_space:
            tokens = sentence.split()
            offsets = None
            if return_offsets:
                offsets = []
                offset = 0
                for tok in tokens:
                    offsets.append((offset, offset + len(tok)))
                    offset += len(tok) + 1
            res = (tokens,)
            if offsets is not None:
                res += (offsets,)
            return res
        subwords = self.base_tokenizer.tokenize(sentence)
        words, subword_offsets = subwordlist_to_wordlist(subwords)
        if not return_offsets:
            return words
        offset_mapping = self.base_tokenizer(sentence, return_offsets_mapping=True)[
            "offset_mapping"
        ]
        offsets = [
            (offset_mapping[elt[0] + 1][0], offset_mapping[elt[1]][1]) for elt in subword_offsets
        ]
        return words, offsets


def load_embedding(
    fname, vocabulary: Optional[Set[str]] = None, limit=None
) -> Tuple[List[str], np.ndarray]:
    with open(fname, "r") as f:
        count, dim = map(int, next(f).strip("\n").split())
        limit = count if limit is None else limit
        words = []
        output = np.zeros(((limit if vocabulary is None else len(vocabulary)), dim))
        for line in f:
            word, vec = line.strip("\n").split(" ", 1)
            vec = np.asarray(np.fromstring(vec, sep=" ", dtype="float"))
            if vocabulary is None or word in vocabulary:
                output[len(words)] = vec
                words.append(word)
            if len(words) == (limit if vocabulary is None else len(vocabulary)):
                break
    return words, output[: len(words)]


def get_metric_fn():
    metric = load_metric("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [p for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [l for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    return compute_metrics


def post_on_slack(message: str, thread_ts=None):
    webhook = os.getenv("SLACK_WEBHOOK")
    channel = os.getenv("SLACK_CHANNEL")
    slack_oauth = os.getenv("SLACK_OAUTH")

    if webhook is None:
        return

    if "requests" not in locals():
        import requests

    if channel is not None and slack_oauth is not None:
        res = requests.post(
            "https://slack.com/api/chat.postMessage",
            json={
                "channel": channel,
                "text": message,
                **({"thread_ts": thread_ts} if thread_ts is not None else {}),
            },
            headers={"Authorization": f"Bearer {slack_oauth}"},
        )
        try:
            result = res.json().get("ts")
        except:
            logging.error(f"slack alert: could not retrieve `ts`: {res.content}")
            result = None
        return result

    requests.post(webhook, json={"text": message})

    return None
