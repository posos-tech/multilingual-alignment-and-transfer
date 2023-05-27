# Multilingual alignment and transfer

This repository holds the code for the following papers:

- [Multilingual Transformer Encoders: a Word-Level Task-Agnostic Evaluation](https://arxiv.org/abs/2207.09076v1), Félix Gaschi, François Plesse, Parisa Rastin, Yannick Toussaint. (IJCNN 2022)

## Architecture of the repository

- `download_resources` contain scripts to download necessary resources
- `multilingual_eval` contain the source code
- `scripts` contains launchables for reproducing experiments

## How to use the repository

The reusable source code is found in `multilingual_eval`, while paper-specific scripts that allows to reproduce a specific experiments and figures from a given paper are found in dedicated subdirectories of `scripts`:

- [scripts/2022_ijcnn](scripts/2022_ijcnn/README.md) for IJCNN 2022

## Runs

XNLI dico (73 pairs): 4blmllpi, 0dfl7bnl, 7c8n1309, 61hr4knm

test on XNLI with true multiparallel: kj8a75ut

test on XQuad with true multiparallel: ye1jps8e
