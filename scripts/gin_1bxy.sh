#!/bin/bash

#SBATCH --job-name=1bxy_gin
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=20G
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err
cd ~/project/ProGSNN-2
module load miniconda
conda activate mfcn

python train_GIN.py --protein "1bxy"