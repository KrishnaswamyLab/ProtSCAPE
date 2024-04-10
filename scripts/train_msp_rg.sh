#!/bin/bash

#SBATCH --job-name=rg_gpu_progsnn
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

python train_progsnn_msp_rg.py