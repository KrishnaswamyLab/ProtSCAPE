#!/bin/bash

# example slurm job scheduling: GB3
# sbatch /scratch/user/u.dj107237/progsnn/Code/aces_job.sh \
# --machine aces \
# --protein_name "BPTI" \
# --ref_traj_pickle_path "/scratch/user/u.dj107237/progsnn/data/reftraj_BPTI.p" \
# --run_index 1 \
# --data_pickle_path "/scratch/user/u.dj107237/progsnn/data/mdfp_dataset_BPTI.p" \
# --save_results_folder "/scratch/user/u.dj107237/progsnn/progsnn/results" \
# --k_cv 5 \
# --min_epochs 10 \
# --max_epochs 100 \
# --batch_size 512 \
# --learn_rate 0.001 \
# --verbosity 0

#SBATCH --job-name=progsnn
#SBATCH --output=/scratch/user/u.dj107237/progsnn/job_outputs/%j
#SBATCH --partition=gpu
#SBATCH --array=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=8000M
#SBATCH --time=00:15:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=davejohnson408@u.boisestate.edu

# turn off multithreading
export OMP_NUM_THREADS=1
# silence tqdm progress bars [any non-empty string will silence]
export TQDM_DISABLE=1
# export project root directory
export PROJ_ROOT=/scratch/user/u.dj107237/progsnn
# set path to script to execute
script=${PROJ_ROOT}/Code/vae_run_cv.py

# grab options passed on command line
while getopts ":a:n:m:f:r:d:s:k:i:x:b:l:v:" opt; do
  case $opt in
    a) machine="$OPTARG"
    ;;
    n) protein_name="$OPTARG"
    ;;
    m) minepochs="$OPTARG"
    ;;
    f) ref_traj_pickle_path="$OPTARG"
    ;;
    r) run_index="$OPTARG"
    ;;
    d) data_pickle_path="$OPTARG"
    ;;
    s) save_results_folder="$OPTARG"
    ;;
    k) k_cv="$OPTARG"
    ;;
    i) min_epochs="$OPTARG"
    ;;
    x) max_epochs="$OPTARG"
    ;;
    b) batch_size="$OPTARG"
    ;;
    l) learn_rate="$OPTARG"
    ;;
    v) verbosity="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

# load conda env and call script, passing args (ACES)
# must load WebProxy to access internet (e.g. download models)
module purge
# module load WebProxy
module load Anaconda3/2022.10
source activate prodigy

python3 $script \
--machine $machine \
--protein_name $protein_name \
--ref_traj_pickle_path $ \
--run_index $run_index \
--data_pickle_path $ \
--save_results_folder $ \
--k_cv $k_cv \
--min_epochs $min_epochs \
--max_epochs $max_epochs \
--batch_size $batch_size \
--learn_rate $learn_rate \
--verbosity $verbosity

