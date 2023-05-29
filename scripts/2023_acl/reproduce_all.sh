#!/bin/bash

set -e

source parameters.sh

langs="ar es fr ru zh"

mkdir -p $DATA_DIR

CACHE_DIR=$DATA_DIR/cache/datasets
OPUS_DIR=$DATA_DIR/opus100
MULTIUN_DIR=$DATA_DIR/multi_un
MUSE_DIR=$DATA_DIR/muse_dictionaries
AWESOME_MODEL_DIR=$DATA_DIR/awesome-without-co
TRANSLATION_DIR=$DATA_DIR/translation
FASTALIGN_DIR=$DATA_DIR/fastalign
DICOALIGN_DIR=$DATA_DIR/dico-align
AWESOME_DIR=$DATA_DIR/awesome-align

mkdir -p $CACHE_DIR
mkdir -p $OPUS_DIR
mkdir -p $MULTIUN_DIR
mkdir -p $MUSE_DIR
mkdir -p $AWESOME_MODEL_DIR
mkdir -p $TRANSLATION_DIR
mkdir -p $FASTALIGN_DIR
mkdir -p $DICOALIGN_DIR
mkdir -p $AWESOME_DIR

export DATA_DIR=$DATA_DIR
export OPUS_DIR=$OPUS_DIR
export MUSE_DIR=$MUSE_DIR
export AWESOME_MODEL_DIR=$AWESOME_MODEL_DIR
export TRANSLATION_DIR=$TRANSLATION_DIR
export FASTALIGN_DIR=$FASTALIGN_DIR
export DICOALIGN_DIR=$DICOALIGN_DIR
export AWESOME_DIR=$AWESOME_DIR

# download muse dictionaries
bash download_resources/muse_dictionaries.sh $MUSE_DIR $langs

# download OPUS 100
bash download_resources/opus100.sh $OPUS_DIR $langs

# download multi-un
bash download_resources/multi_un.sh $MULTIUN_DIR

# Sample 1B for each lang pair of multi-un
for lang in $langs; do

    pair=$(python -c "print('-'.join(sorted(['en', '$lang'])))")

    if [ ! -f $MULTIUN_DIR/$pair/$pair.short.$lang ]; then
        echo "Sampling 1B sentences from multi-un in $pair"
        python subscripts/reservoir_sampling.py \
            $MULTIUN_DIR/$pair/UNv1.0.$pair.$lang \
            $MULTIUN_DIR/$pair/UNv1.0.$pair.en \
            --output_files \
            $MULTIUN_DIR/$pair/$pair.short.$lang \
            $MULTIUN_DIR/$pair/$pair.short.en \
            --num_samples 1000000
    fi

    # To make some room 
    rm $MULTIUN_DIR/$pair/UNv1.0.$pair.*

done


# install fastalign
bash download_resources/fastalign.sh

# install Stanford segmenter for Chinese
bash download_resources/stanford_tokenizer.sh

# installe awesome-align and download model without-co
bash download_resources/awesome_align.sh  $AWESOME_MODEL_DIR

# Create virtualenv for awesome-align
if [ ! -d $DATA_DIR/venvs/awesome-align ]; then
    python -m venv $DATA_DIR/venvs/awesome-align

    cd tools/awesome-align
    source $DATA_DIR/venvs/awesome-align/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    python setup.py install
    deactivate
    cd ../..
fi


exit 0


for lang in $langs; do
    echo "parsing lang $lang for opus-100"

    mkdir -p $TRANSLATION_DIR/opus100
    mkdir -p $DICOALIGN_DIR/opus100
    mkdir -p $AWESOME_DIR/opus100

    pair=$(python -c "print('-'.join(sorted(['en', '$lang'])))")

    # Create FastAlign-compatible tokenized translation dataset
    python subscripts/prepare_pharaoh_dataset.py \
        $OPUS_DIR/$pair/opus.$pair-train.en \
        $OPUS_DIR/$pair/opus.$pair-train.$lang \
        $TRANSLATION_DIR/opus100/en-$lang.tokenized.train.txt \
        --left_lang en --right_lang $lang


    # Align with FastAlign
    if [ ! -f $FASTALIGN_DIR/opus100/en-$lang.train ]; then
        mkdir -p $FASTALIGN_DIR/opus100/fastalign/en-$lang
        bash scripts/25_fast_align.sh $TRANSLATION_DIR/opus100/en-$lang.tokenized.train.txt $FASTALIGN_DIR/opus100/fastalign/en-$lang
        mv $FASTALIGN_DIR/opus100/fastalign/en-$lang/symmetrized.align $FASTALIGN_DIR/opus100/en-$lang.train
    fi

    # Align with bilingual dictionaries
    if [ ! -f $DICOALIGN_DIR/opus100/en-$lang.train ]; then
        python scripts/32_dico_align.py $TRANSLATION_DIR/opus100/en-$lang.tokenized.train.txt $MUSE_DIR en $lang $DICOALIGN_DIR/opus100/en-$lang.train
    fi

    # Align with AWESOME-align
    if [ ! -f $AWESOME_DIR/opus100/en-$lang.train ]; then
        bash scripts/33_awesome_align.sh $TRANSLATION_DIR/opus100/en-$lang.tokenized.train.txt $AWESOME_DIR/opus100/en-$lang.train $AWESOME_MODEL_DIR/model_without_co $DATA_DIR
    fi

done

for lang in $multi_un_langs; do
    echo "parsing lang $lang for multi-un"

    mkdir -p $TRANSLATION_DIR/multi_un
    mkdir -p $DICOALIGN_DIR/multi_un
    mkdir -p $AWESOME_DIR/multi_un

    pair=$(python -c "print('-'.join(sorted(['en', '$lang'])))")

    # Create FastAlign-compatible tokenized translation dataset
    python subscripts/prepare_pharaoh_dataset.py \
        $OPUS_DIR/$pair/$pair.short.en \
        $OPUS_DIR/$pair/$pair.short.$lang \
        $TRANSLATION_DIR/multi_un/en-$lang.tokenized.train.txt \
        --left_lang en --right_lang $lang

    # Align with FastAlign
    if [ ! -f $FASTALIGN_DIR/multi_un/en-$lang.train ]; then
        mkdir -p $FASTALIGN_DIR/multi_un/fastalign/en-$lang
        bash scripts/25_fast_align.sh $TRANSLATION_DIR/multi_un/en-$lang.tokenized.train.txt $FASTALIGN_DIR/multi_un/fastalign/en-$lang
        mv $FASTALIGN_DIR/multi_un/fastalign/en-$lang/symmetrized.align $FASTALIGN_DIR/multi_un/en-$lang.train
    fi

    # Align with bilingual dictionaries
    if [ ! -f $DICOALIGN_DIR/multi_un/en-$lang.train ]; then
        python scripts/32_dico_align.py $TRANSLATION_DIR/multi_un/en-$lang.tokenized.train.txt $MUSE_DIR en $lang $DICOALIGN_DIR/multi_un/en-$lang.train
    fi

    # Align with AWESOME-align
    if [ ! -f $AWESOME_DIR/multi_un/en-$lang.train ]; then
        bash scripts/33_awesome_align.sh $TRANSLATION_DIR/multi_un/en-$lang.tokenized.train.txt $AWESOME_DIR/multi_un/en-$lang.train $AWESOME_MODEL_DIR/model_without_co $DATA_DIR
    fi
done