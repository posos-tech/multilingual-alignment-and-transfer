
set -e

INPUT_FILE=$1
OUTPUT_FILE=$2
MODEL_PATH=$3
DATA_DIR=$4

source $DATA_DIR/venvs/awesome-align/bin/activate

CUDA_VISIBLE_DEVICES=0 awesome-align \
    --output_file=$OUTPUT_FILE \
    --model_name_or_path=$MODEL_PATH \
    --data_file=$INPUT_FILE \
    --extraction 'softmax' \
    --batch_size 32

deactivate