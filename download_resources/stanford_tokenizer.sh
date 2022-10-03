set -e

mkdir -p tools

cd tools

if [ ! -d stanford-corenlp-full-2016-10-31 ]; then
    wget http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip
    unzip stanford-corenlp-full-2016-10-31.zip
fi

cd stanford-corenlp-full-2016-10-31

wget http://nlp.stanford.edu/software/stanford-chinese-corenlp-2016-10-31-models.jar
wget https://raw.githubusercontent.com/stanfordnlp/CoreNLP/master/src/edu/stanford/nlp/pipeline/StanfordCoreNLP-chinese.properties 

cd ../..