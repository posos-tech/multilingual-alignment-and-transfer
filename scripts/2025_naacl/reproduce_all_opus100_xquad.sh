#!/bin/bash

set -e

DATA_DIR=$1
DATASET=$2
MODEL=$3
ADD_ARGS=$4

langs="ar de el es hi ro ru th tr vi zh"
additional_langs="bg cs lv af ca da fa fi fr he hu it ja ko lt no pl pt sk sl sv ta uk"

mkdir -p $DATA_DIR

CACHE_DIR=$DATA_DIR/cache/datasets
TRANSLATION_DIR=$DATA_DIR/translation
FASTALIGN_DIR=$DATA_DIR/fastalign
DICOALIGN_DIR=$DATA_DIR/dico-align
AWESOME_DIR=$DATA_DIR/awesome-align
RESULT_DIR=$DATA_DIR/raw_results

mkdir -p $CACHE_DIR
mkdir -p $TRANSLATION_DIR
mkdir -p $FASTALIGN_DIR
mkdir -p $DICOALIGN_DIR
mkdir -p $AWESOME_DIR
mkdir -p $RESULT_DIR

export DATA_DIR=$DATA_DIR
export TRANSLATION_DIR=$TRANSLATION_DIR
export FASTALIGN_DIR=$FASTALIGN_DIR
export DICOALIGN_DIR=$DICOALIGN_DIR
export AWESOME_DIR=$AWESOME_DIR


################
# BASELINES
################


python scripts/2023_acl/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/$DATASET \
    --fastalign_dir $FASTALIGN_DIR/$DATASET \
    --dico_dir $DICOALIGN_DIR/$DATASET \
    --awesome_dir $AWESOME_DIR/$DATASET \
    --strategies baseline \
    --models $MODEL \
    --tasks xquad \
    --cache_dir $CACHE_DIR \
    --n_epochs 5 \
    --right_langs $langs \
    --additional_realignment_langs $additional_langs \
    --output_file $RESULT_DIR/${MODEL}__${DATASET}__xquad.csv $ADD_ARGS


python scripts/2023_acl/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/$DATASET \
    --fastalign_dir $FASTALIGN_DIR/$DATASET \
    --dico_dir $DICOALIGN_DIR/$DATASET \
    --awesome_dir $AWESOME_DIR/$DATASET \
    --strategies before_dico \
    --models $MODEL \
    --tasks xquad \
    --cache_dir $CACHE_DIR \
    --n_epochs 5 \
    --right_langs $langs \
    --additional_realignment_langs $additional_langs \
    --output_file $RESULT_DIR/${MODEL}__${DATASET}__xquad.csv $ADD_ARGS


###########################################
# PARTIAL-REALIGNMENT BEFORE - FRONT FROZEN
###########################################

python scripts/2023_acl/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/$DATASET \
    --fastalign_dir $FASTALIGN_DIR/$DATASET \
    --dico_dir $DICOALIGN_DIR/$DATASET \
    --awesome_dir $AWESOME_DIR/$DATASET \
    --strategies freeze_realign_unfreeze_dico \
    --models $MODEL \
    --tasks xquad \
    --cache_dir $CACHE_DIR \
    --n_epochs 5 \
    --right_langs $langs \
    --additional_realignment_langs $additional_langs \
    --output_file $RESULT_DIR/${MODEL}__${DATASET}__xquad.csv $ADD_ARGS


###########################################
# PARTIAL-REALIGNMENT BEFORE - BACK FROZEN
###########################################

python scripts/2023_acl/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/$DATASET \
    --fastalign_dir $FASTALIGN_DIR/$DATASET \
    --dico_dir $DICOALIGN_DIR/$DATASET \
    --awesome_dir $AWESOME_DIR/$DATASET \
    --strategies freeze_realign_unfreeze_last_half_dico \
    --models $MODEL \
    --tasks xquad \
    --cache_dir $CACHE_DIR \
    --n_epochs 5 \
    --right_langs $langs \
    --additional_realignment_langs $additional_langs \
    --output_file $RESULT_DIR/${MODEL}__${DATASET}__xquad.csv $ADD_ARGS