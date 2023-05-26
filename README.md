# Multilingual alignment and transfer

This repository holds the code for the following papers:

- [Multilingual Transformer Encoders: a Word-Level Task-Agnostic Evaluation](https://arxiv.org/abs/2207.09076v1), Félix Gaschi, François Plesse, Parisa Rastin, Yannick Toussaint. IJCNN 2022
- Improving Cross-lingual Transfer With Realignment as an Auxiliary Task (Work-in-progress and tentative title)

## Architecture of the repository

- `download_resources` contain scripts to download necessary resources
- `multilingual_eval` contain the source code
- `scripts` contains launchables (mainly Python) for obtaining raw results of experiments
- `generate_figures` contains launchables for obtaining figures and LaTeX tables from the raw results 
- `reproduce_results` contains requirements files and scripts to reproduce all the results of a whole paper

## How to use the repository

### To reproduce a single experiment

install the relevant dependencies (ideally in a virtual environment):

- from the file reproduce_results/ijcnn_2022_requirements.txt for IJCNN experiments (scripts 01 and 02)
- from the file reproduce_results/eacl_2023_requirements.txt for WIP submission to EACL (scripts 03-06,10,12,13,18-23)

Download the required data:

- FastText aligned embedding (01 and 02) using `bash download_resources/aligned_fasttext.sh $OUTPUT_DIR "en fr zh"` (in the example download aligned french, english and chinese embedding in $OUTPUT_DIR, but you might need more embeddings)
- MUSE dictionaries (most scripts) using `bash download_resources/muse_dictionaries.sh $OUTPUT_DIR "fr zh"` (in the example download full dictionaries for en-fr, fr-en, en-zh and zh-en pairs, but you might need more dictionaries)

Then you can launch the script you want in `scripts` (paying attention to the required arguments).

### To reproduce all the results from a paper

Create a `parameters.sh` file and fill according to the template provided in `sample_parameters.sh`

Install the dependencies from the corresponding requirements that can be found in `reproduce_results`

Then launch the script to reproduce the results which will be found in `reproduce_results` which is referenced by conference and year.

## Runs

XNLI dico (73 pairs): 4blmllpi, 0dfl7bnl, 7c8n1309, 61hr4knm

test on XNLI with true multiparallel: kj8a75ut

test on XQuad with true multiparallel: ye1jps8e
