
set -e

INPUT_FILE=$1
OUTPUT_DIR=$2

mkdir -p $OUTPUT_DIR

./tools/fast_align/build/fast_align -i $INPUT_FILE -d -o -v > $OUTPUT_DIR/forward.align
./tools/fast_align/build/fast_align -i $INPUT_FILE -d -o -v -r > $OUTPUT_DIR/reverse.align
./tools/fast_align/build/atools -i $OUTPUT_DIR/forward.align -j $OUTPUT_DIR/reverse.align -c grow-diag-final-and > $OUTPUT_DIR/symmetrized.align
