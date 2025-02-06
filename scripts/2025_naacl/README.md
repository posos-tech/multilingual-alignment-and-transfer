# ALIGNFREEZE: Navigating the Impact of Realignment on the Layers of Multilingual Models Across Diverse Languages

Code for the paper: ALIGNFREEZE: Navigating the Impact of Realignment on the Layers of Multilingual Models Across Diverse Languages. Steve Bakos, Félix Gaschi, David Guzmán, Riddhi More, Kelly Chutong Li, En-Shiun Annie Lee. NAACL 2025 (link coming soon)

This directory contains the code for experiments for the above-mentioned paper. It relies mainly on the code from a previous paper (see [scripts/2023_acl](scripts/2023_acl/README.md)). 

To perform the partial-freezing method presented in the paper, two new realignment strategies were added:

- freeze_realign_unfreeze_<ALIGNER> which freezes the first half of the model during realignment
- freeze_realign_unfreeze_last_half_<ALIGNER> which freezes the upper half of the model during realignment

This additional strategy can directly be used with the scripts provided in `scripts/2023_acl`.

## Requirements

This was tested in Python 3.9, with requirements provided in `scripts/2023_acl/requirements.txt` and some additional requirements in `scripts/2025_naacl/additional_requirements.txt`, where `wandb` is actually optional and `brokenaxes` is only useful for generating figures.

Some additional data and tools are required (similarly to 2023_acl):

- MUSE dictionaries (required for the *_dico strategies)
- The OPUS 100 dataset (required for any realignment method)
- fastalign (*_fastalign strategies)
- AWESOME-align along with some model (*_awesome strategies)
- the Stanford tokenizer (required for Chinese)

The script `scripts/2025_naacl/download.sh` should allow to download everything necessary and to precompute the alignment pairs for the realignment training. This can take quite long to run. You can reduce the number of languages or aligners involved to obtain it faster.

## How to use?

This directory then only contains bash scripts to reproduce the entire experiments produced in the paper. For example, to reproduce experiments on PoS-tagging, you can launch the following script:

```
bash scripts/2025_naacl/reproduce_all_opus100_udpos.sh <DATA_DIR> opus100 bert-base-multilingual-cased
```

`<DATA_DIR>` must be the same path that you used in the `download.sh` script.

## Additional content