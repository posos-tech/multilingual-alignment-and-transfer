#!/bin/bash

set -e


OUTPUT_DIR=$1

langs=$2

for lang in $langs; do
    if [ ! -f $OUTPUT_DIR/en-$lang.txt ]; then
        wget https://dl.fbaipublicfiles.com/arrival/dictionaries/en-$lang.txt -O $OUTPUT_DIR/en-$lang.txt
    fi
    if [ ! -f $OUTPUT_DIR/$lang-en.txt ]; then
        wget https://dl.fbaipublicfiles.com/arrival/dictionaries/en-$lang.txt -O $OUTPUT_DIR/$lang-en.txt
    fi
done