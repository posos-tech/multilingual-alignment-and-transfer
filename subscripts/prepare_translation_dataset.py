"""
Tokenize OPUS100 dataset and produce a FastAlign-compatile output 
"""


import os
import sys
from typing import Optional
from contextlib import ExitStack
import logging

sys.path.append(os.curdir)

from multilingual_eval.tokenization.chinese_segmenter import StanfordSegmenter
from multilingual_eval.utils import RegexTokenizer, ChineseTokenizer
from multilingual_eval.utils import count_lines

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("left_lang")
    parser.add_argument("right_lang")
    parser.add_argument("input_dir")
    parser.add_argument("output_file")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", dest="overwrite")
    parser.add_argument(
        "--no-sort",
        action="store_false",
        dest="sort",
        help="to use if left_lang and right_lang must not be sorted alphabetically in the files prefix",
    )
    parser.add_argument("--filename_template", type=str, default="opus.%s-%s-train.%s")
    parser.set_defaults(overwrite=False, sort=True)
    args = parser.parse_args()

    left_lang = args.left_lang
    right_lang = args.right_lang

    first_lang, second_lang = (
        (left_lang, right_lang)
        if not args.sort or left_lang < right_lang
        else (right_lang, left_lang)
    )

    left_lang_file = os.path.join(
        args.input_dir,
        f"{first_lang}-{second_lang}",
        args.filename_template % (first_lang, second_lang, left_lang),
    )
    right_lang_file = os.path.join(
        args.input_dir,
        f"{first_lang}-{second_lang}",
        args.filename_template % (first_lang, second_lang, right_lang),
    )

    # Compute number of lines already parsed if not overriding
    if not args.overwrite and os.path.isfile(args.output_file):
        lines_to_parse = count_lines(left_lang_file)
        lines_parsed = count_lines(args.output_file)

        start_line = lines_to_parse if lines_parsed >= lines_to_parse else lines_parsed
    else:
        start_line = 0

    with ExitStack() as stack:
        if "zh" in [left_lang, right_lang]:
            zh_segmenter = stack.enter_context(StanfordSegmenter())
        left_reader = stack.enter_context(open(left_lang_file, "r"))
        right_reader = stack.enter_context(open(right_lang_file, "r"))
        writer = stack.enter_context(open(args.output_file, "w" if args.overwrite else "a"))

        logging.info(f"Skipping {start_line} lines")
        for _ in range(start_line):
            _ = next(left_reader)
            _ = next(right_reader)

        left_tokenizer = ChineseTokenizer(zh_segmenter) if left_lang == "zh" else RegexTokenizer()
        right_tokenizer = ChineseTokenizer(zh_segmenter) if right_lang == "zh" else RegexTokenizer()

        for left_line, right_line in zip(left_reader, right_reader):
            left_tokens = left_tokenizer.tokenize(left_line)
            right_tokens = right_tokenizer.tokenize(right_line)

            writer.write(" ".join(left_tokens) + " ||| " + " ".join(right_tokens) + "\n")
