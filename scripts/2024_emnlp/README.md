# ALIGNFREEZE: Navigating the Impact of Realignment on the Layers of Multilingual Models Across Diverse Languages

Code for the paper: ALIGNFREEZE: Navigating the Impact of Realignment on the Layers of Multilingual Models Across Diverse Languages. Steve Bakos, Félix Gaschi, David Guzmán, Riddhi More, Kelly Chutong Li, En-Shiun Annie Lee. NAACL 2025 (link coming soon)

This directory contains the code for experiments for the above-mentioned paper. It relies mainly on the code from a previous paper (see [scripts/2023_acl](scripts/2023_acl/README.md)). 

To perform the partial-freezing method presented in the paper, two new realignment strategies were added:

- freeze_realign_unfreeze_<ALIGNER> which freezes the first half of the model during realignment
- freeze_realign_unfreeze_last_half_<ALIGNER> which freezes the upper half of the model during realignment

This additional strategy can directly be used with the scripts provided in `scripts/2023_acl`.

This directory then only contains bash scripts to reproduce the entire experiments produced in the paper:

- 
