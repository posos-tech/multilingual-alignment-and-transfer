
set -e

MODEL_DIR=$1

mkdir -p $MODEL_DIR

if [ ! -d venvs/awesome-align ]; then
    python -m venv venvs/awesome-align
fi

cd tools

if [ ! -d awesome-align ]; then
    git clone git@github.com:neulab/awesome-align.git
    cd awesome-align

    source ../../venvs/awesome-align/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    python setup.py install

    deactivate

    cd ..
fi

cd ..

if [ ! -d $MODEL_DIR/model_without_co ]; then
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1IcQx6t5qtv4bdcGjjVCwXnRkpr67eisJ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1IcQx6t5qtv4bdcGjjVCwXnRkpr67eisJ" -O $MODEL_DIR/awesome_without_co.zip && rm -rf /tmp/cookies.txt
    unzip -d $MODEL_DIR $MODEL_DIR/awesome_without_co.zip
fi