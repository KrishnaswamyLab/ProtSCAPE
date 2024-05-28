# ProtSCAPE: Protein Transformer with Scattering, Attention and Positional Embeddng

# Introduction

ProtSCAPE utilizes the learnable geometric scattering transform together with transformer-based attention mechanisms to capture and interpolate protein dynamics from molecular dynamics (MD) simulations. ProtSCAPE utilizes the multi-scale nature of the geometric scattering transform to extract features from protein structures conceptualized as graphs. It then integrates these features with dual attention structures, which focus on the residues and amino acid signals respectively, to generate latent representations of protein trajectories. Furthermore, ProtSCAPE incorporates a regression head to generate a structured, temporally coherent latent space, facilitating the accurate interpolation of protein conformations.

![Project Logo](images/Schematic.png)

# Dependencies

ProtSCAPE requires depedencies listed in the `protscape.yml` file. In order to install the dependencies, run the following command on your machine:

```sh
conda env create -f protscape.yml
```

Once the conda environment `protscape` has been created, the following command must be run in order to activate it:

```sh
conda activate protscape
```

# Quick Start

In order to train and test the ATLAS and Deshaw datasets on five fold cross validation, run the following commands respectively:

```sh
python atlas_five_fold.py --protein "protein_name"
python deshaw_five_fold.py --protein "protein_name"
```




