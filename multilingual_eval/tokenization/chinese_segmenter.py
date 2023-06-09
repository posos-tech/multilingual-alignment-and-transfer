import requests
from time import sleep
import os
from subprocess import Popen, PIPE, DEVNULL
from nltk.parse.corenlp import CoreNLPParser
import threading
import sys
import logging


def filtered_stderr_logger(p):
    for line in p.stderr:
        text = line.decode()
        if len(text.split()) > 2 and text.split()[1] == "INFO":
            continue
        print(text, file=sys.stderr)


class StanfordSegmenter:
    def __init__(self):
        self.segmenter = None
        self.server_process = None
        self.entered = False

    def __enter__(self):
        self.entered = True
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        self.entered = False

    def __call__(self, sent):
        if not self.entered:
            raise Exception(
                f"LasyChineseSegmenter should be used inside a with statement for necessary cleanup"
            )

        splitting_char = "。"
        parts = sent.strip().split(splitting_char)
        tokens = []
        for i, part in enumerate(parts):
            if i > 0:
                tokens.append(splitting_char)
            if len(part.strip()) == 0:
                continue
            try:
                result = self.segmenter.api_call(part, {"annotators": "tokenize,ssplit"})
            except requests.exceptions.HTTPError as e:
                logging.error(f"Got HTTPError with following sentence: {repr(part)}")
                raise e
            tokens += [
                token["originalText"] or token["word"]
                for sentence in result["sentences"]
                for token in sentence["tokens"]
            ]

        return tokens

    def initialize(self):
        if self.segmenter is None:
            if not os.path.isdir("tools/stanford-corenlp-full-2016-10-31"):
                raise Exception(
                    f"tools/stanford-corenlp-full-2016-10-31 does not exist, please install the Stanford Segmenter (download_resources/stanford_tokenizer.sh)"
                )
            self.server_process = Popen(
                ["/bin/bash", "subscripts/launch_corenlp_server.sh"], stderr=DEVNULL, stdout=DEVNULL
            )

            self.segmenter = CoreNLPParser("http://localhost:9001", encoding="utf8")
            sleep(2)
            self.start_and_wait_for_availability()

    def start_and_wait_for_availability(self, max_iter=8, wait=2):
        if max_iter == 0:
            raise Exception("Max iteration exceeded for waiting for Stanford Segmenter to start")
        try:
            self.segmenter.api_call(
                "只是教授和警察双方都对不尊重的暗示表现得过于敏感", {"annotators": "tokenize,ssplit"}, timeout=1
            )
        except (requests.exceptions.ConnectionError, ConnectionRefusedError) as e:
            sleep(wait)
            self.start_and_wait_for_availability(max_iter=max_iter - 1, wait=wait * 2)

    def cleanup(self):
        if self.server_process is not None:
            self.server_process.terminate()

        self.segmenter = None
