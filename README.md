# Multilingual alignment and transfer

This repository holds the code for the following papers:

- [Multilingual Transformer Encoders: a Word-Level Task-Agnostic Evaluation](https://arxiv.org/abs/2207.09076v1), Félix Gaschi, François Plesse, Parisa Rastin, Yannick Toussaint. IJCNN 2022
- Improving Cross-lingual Transfer With Realignment as an Auxiliary Task (Work-in-progress and tentative title) [overleaf link](https://www.overleaf.com/read/qrbqbbrmbrsx)

## Architecture of the repository

- `download_resources` contain scripts to download necessary resources
- `multilingual_eval` contain the source code
- `scripts` contains launchables (mainly Python) for obtaining raw results of experiments
- `generate_figures` contains launchables for obtaining figures and LaTeX tables from the raw results 
- `reproduce_results` contains requirements files and scripts to reproduce all the results of a whole paper

## How to use the repository

### Focus on experiments from the current branch

Work in progress for the EACL submission can be found in the following scripts:

- scripts/21_realignment_for_pos.py
- scripts/22_realignment_for_ner.py
- scripts/23_realignment_for_xnli.py

Requirements are written in `reproduce_results/eacl_2023_requirements.txt` and can be installed in a virtual environment

For those to work, you need to download bilingual dictionaries for the following languages: `["ar", "es", "fr", "ru", "zh"]`

```{bash}
bash $DICO_PATH "ar es fr ru zh"
```

Then you can use any of the above-mentionned script by doing the following (example with the 21st):

```{bash}
python scripts/21_realignment_for_pos.py $DICO_PATH
```

Additional parameters:

- to juste test the script without waiting too much, you can add `--debug`
- to store the datasets which will be loaded from Huggingface in a specific directory, you can add `--data_dir <YOUR_SPECIFIC_DIRECTORY>`
- those scripts are grid search using wandb, if you want to create a new agent for an existing grid search instead of starting a new one, you just need the id of the sweep and add `--sweep_id <ID_OF_THE_SWEEP>`

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

## Inventory of the runs on wandb

### scripts/21_realignment_for_pos.py

- [x] s4f3ngqk (finished): grid search mBERT
- [x] wy9tsfd9 (finished): grid-search XLM-R base
- [x] b452skly (finished): XLM-R Large baseline
- [x] i17jod5q (finished): mBERT baseline
- [x] 46egpf24 (finished): XLM-R Base baseline

### scripts/22_realignment_for_ner.py

- [x] ta7mva4z (finished): NER XLM-R Large baseline
- [x] kdybfi7l (finished): NER mBERT baseline
- [x] 7wjim0zy (finished): NER XLM-R Base baseline
- [x] 0tl6jkpq (finished): NER mBERT Base grid-search
- [x] ixp38wf8 (finished): NER XLM-R Base grid-search

### scripts/23_realignment_for_xnli.py

- [ ] hf252z4e (unfinished - to resume): XNLI XLM-R Large
