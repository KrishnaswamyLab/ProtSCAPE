"""
Protein Dynamics Interpolation via a Geometric Scattering Transformer
Get CV metrics for VAE comparison model

to debug, enter in shell: 
    export PYTHONFAULTHANDLER=1
to revert to system python:
    export PATH="/Library/Frameworks/Python.framework/Versions/3.11/bin:$PATH"
to
    export PYTHONPATH="/Library/Frameworks/Python.framework/Versions/3.11/bin/python3"

Example calls:

>> to run 5-fold CV and calc metrics, run with
    --k_cv 5
    --calc_metrics T

>> to just get latent space embeddings, run with 
    --k_cv 1
    --calc_metrics F


-----------------
on local machine:
-----------------
python3 <YOUR_PATH>/vae_run_cv.py \
--machine local \
--protein_name BPTI \
--ref_traj_pickle_path "<YOUR_PATH>/data/pickled_datasets/reference_trajectories/reftraj_BPTI.p" \
--run_index 1 \
--data_pickle_path "<YOUR_PATH>/data/pickled_datasets/unscaled/mdfp_dataset_BPTI.p" \
--save_results_folder "<YOUR_PATH>/results" \
--k_cv 5 \
--calc_metrics T \
--min_epochs 99 \
--max_epochs 100 \
--batch_size 512 \
--learn_rate 0.001 \
--verbosity 1

"""

print('Running...')

"""
ENV
"""

# hard-coded constants
FOLD_TO_SAVE = 0
METRICS = ('scc', 'pcc', 'rmsd', 'dope')
SAVE_LS_EMBEDDINGS = True
# identifies the target name in the input_dict['target'] subdict
TARGET_NAME = 'coords'

import sys
sys.path.insert(0, '../')
import os
import time
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mdtraj as md

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--machine', default=None, type=str, 
                    help='name of machine on which job is run')
parser.add_argument('-n', '--protein_name', default=None, type=str, 
                    help='name of protein')
parser.add_argument('-f', '--ref_traj_pickle_path', 
                    default=None, type=str, 
                    help='folder path for ATLAS protein')
parser.add_argument('-r', '--run_index', 
                    default='1', type=int, 
                    help='ATLAS MD simulation run index (1-3)')
parser.add_argument('-d', '--data_pickle_path', default=None, type=str, # os.path.relpath
                    help='path to dataset pickle file')
# parser.add_argument('-t', '--transform_info_filepath', default=None, type=str, 
#                     help='path to dataset transform (scaling) info dictionary pickle file')
parser.add_argument('-s', '--save_results_folder', default=None, type=str, 
                    help='path to folder to save results pickle')
parser.add_argument('-k', '--k_cv', default='5', type=int, 
                    help='number of cross-validation folds')
parser.add_argument('-c', '--calc_metrics', default='T', type=str, 
                    help='calculate validation metrics? (default: True)')
parser.add_argument('-i', '--min_epochs', default='0', type=int,
                    help='min num of train epochs (default: 0)')
parser.add_argument('-x', '--max_epochs', default='25', type=int,
                    help='max num of train epochs (default: 25)')
parser.add_argument('-b', '--batch_size', default='64', type=int,
                    help='batch size (default: 64)')
parser.add_argument('-l', '--learn_rate', default='0.001', type=float,
                    help='optimizer learning rate (default: 0.001)')
parser.add_argument('-v', '--verbosity', default='0', type=int,
                    help='controls console printing during training')
clargs = parser.parse_args()
CALC_METRICS = 't' in clargs.calc_metrics.lower()

# import own modules
import utilities as U
from data_processing import data_utilities as DU
import TrainArgs as TA
from models.VAE import VAE
from train_fn import train_model
import data_processing.Dataset_subclasses
sys.modules['Dataset_subclasses'] = data_processing.Dataset_subclasses
from metrics.metrics_fns import (
    convert_batch_coords,
    EuclideanDistanceCorrs,
    ProteinConfigMetricsEstimates
)

# override some defaults in `TrainArgs.py`.
args = TA.TrainArgs(
    MACHINE=clargs.machine,
    BURNIN_N_EPOCHS=clargs.min_epochs,
    N_EPOCHS=clargs.max_epochs,
    BATCH_SIZE=clargs.batch_size,
    LEARN_RATE=clargs.learn_rate,
    MODEL_SAVE_SUBDIR='latent_space_runs' # puts all model folders of cv in a parent dir
)
SEED_DICT = {
    'GB3': args.CV_SEED_GB3,
    'BPTI': args.CV_SEED_BPTI,
    'UBIQ': args.CV_SEED_UBIQ,
    '1bxy_A': args.CV_SEED_1bxy,
    '1bx7_A': args.CV_SEED_1bx7,
    '1ptq_A': args.CV_SEED_1ptq,
}


# make sure model save dir exists
os.makedirs(args.MODEL_SAVE_DIR, exist_ok=True)

# pickle training args
U.pickle_obj(path=f"{args.MODEL_SAVE_DIR}/train_args.p", obj=args)

# if this is a slurm job:
# stamp print log file with slurm job id
SLURM_JOB_ID = os.environ.get('SLURM_JOB_ID')
if SLURM_JOB_ID is not None:
    jobid_line = f"slurm job id: {SLURM_JOB_ID}"
    with open(args.PRINT_DIR, 'a') as f:
        f.write(jobid_line + "\n")

"""
DATA
"""
# dataset pickle
with open(f"{clargs.data_pickle_path}", "rb") as f:
    dataset = pickle.load(f)
    input_dim = dataset[0]['x'].shape[0]
    # output dim is n_residue * 3 (unrolled coords)
    output_dim = dataset[0]['target'][TARGET_NAME].shape[0]

# OPTIONAL: coords transform/scaling info
# with open(clargs.transform_info_filepath, "rb") as f:
#     transform_info = pickle.load(f)

# load saved reference trajectory pickle
with open(f"{clargs.ref_traj_pickle_path}", "rb") as f:
    ref_traj = pickle.load(f)

# or: load from an ATLAS dataset folder (uses pdb and xtc files)
# ref_traj = md.load(
#     f"{clargs.ref_traj_pickle_path}/{clargs.protein_name}_R{clargs.run_index}.xtc",
#     top=f"{clargs.ref_traj_pickle_path}/{clargs.protein_name}.pdb"
# )



"""
METRICS
# NOTE
# if datasets were rescaled, convert batch coords back to 
# the original scale (angstroms), by passing `transform_info`

NOTE: Use the first frame as a template to back-calculate atomic coords
        from predicted residue coords. (If you use the ground truth frame 
        from the same timestep to back-calculate atomic coords from pred.
        residue coords, you're adding atomic info the model didn't learn,
        and biasing the predicted atomic coords to be more correct.)
Hence, ref_frame = ref_traj[0] (we use the first frame in the traj as ref.)
"""
ref_frame = ref_traj[0]

EDCorrs = EuclideanDistanceCorrs(
    square_dists=False, 
    reduction='mean'
)
dope_kwargs = {
    'multiproc': False, # doesn't work yet...and probably not nec. at our scale
    'multiproc_kwargs': {'n_proc': 4, 'context': 'spawn'},
    'normalize': True,
    'refine': False, # default: False
    'tmp_pdb_savepath': args.DATA_DIR,
    'verbosity': clargs.verbosity
}
PCME = ProteinConfigMetricsEstimates(
    ref_frame=ref_frame, 
    transform_info=None,
    dope_kwargs=dope_kwargs,
    verbosity = clargs.verbosity
)


"""
TRAINING
"""
# patch: if running one fold, still do 80/20 splits
k = clargs.k_cv if clargs.k_cv > 1 else 5
# get valid set indexes list for cross-validations
cv_idx_l = DU.get_cv_idx_l(
    seed=SEED_DICT[clargs.protein_name],
    dataset_size=len(dataset), 
    k=k
)

# set loss function parameters
loss_fn_kwargs = {
    'target_name': TARGET_NAME,
    'rescale_wts_relatively': False, 
    'KLD_wt': 1.0
}

print('Beginning cross-validation...')
time_0 = time.time()
    
def run_fold(i, valid_metrics_dict):
    out = f'\nFold {i}'
    U.print_and_log(out, args.PRINT_DIR)

    # split the full dataset into train and valid sets
    train_data, valid_data = DU.get_train_valid_sets(
        dataset=dataset, 
        valid_idx=cv_idx_l[i]
    )
    datasets = {
        'train': train_data, 
        'valid': valid_data
    }

    # put train and valid sets into dataloaders
    dataloaders = U.get_dataloaders(
        datasets=datasets, 
        seed=args.DATALOADER_RS,
        batch_size=args.BATCH_SIZE,
        drop_last=args.DATALOADER_DROP_LAST,
        num_workers=args.DATALOADER_N_WORKERS,
        pin_memory=(args.ON_CPU == False)
    )
    
    # init new model for each fold
    model = VAE.VAE(
        loss_fn=VAE.vae_loss,
        loss_fn_kwargs=loss_fn_kwargs,
        input_dim=input_dim,
        output_dim=output_dim, 
        encoder_dim_arr=args.ENCODER_DIM_ARR,
        latent_dims=args.LATENT_SPACE_DIMS,
        decoder_dim_arr=args.DECODER_DIM_ARR,
        nonlin_fn=nn.LeakyReLU,
        nonlin_fn_kwargs={'negative_slope': args.RELU_NSLOPE},
        wt_init_fn=nn.init.kaiming_uniform_,
        decoder_final_nonlin_fn=nn.Sigmoid
    )

    # init new optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.LEARN_RATE,
        betas=args.ADAM_BETAS
    )
    
    # train the model for the current fold
    args.MODEL_NAME = f'vae_{i}'
    trained_model, records = train_model(
        args,
        model,
        dataloaders,
        optimizer,
        stop_rule=None, # 'no_improvement'
        save_states=(i == FOLD_TO_SAVE), # save only on first fold
        return_objs=True,
        verbosity=clargs.verbosity
    )

    if CALC_METRICS:
        # get trained model's predictions on the valid set
        trained_model.eval()
        with torch.set_grad_enabled(False):
            for input_dict in dataloaders['valid']:
                # get trained model output (calls `model.forward()`)
                output_dict = trained_model(input_dict) 
    
                # grab preds, targets, and ids
                preds = output_dict['x_hat']
                target = input_dict['target'][TARGET_NAME]
                timestep_ids = input_dict['target']['id'].numpy()
    
                # convert batch coords to correct shape for metrics fns
                pred_coords = convert_batch_coords(preds)
                target_coords = convert_batch_coords(target)
    
                # print('pred_coords\n:', pred_coords)
                # print('target_coords\n:', target_coords)
                
                # update metrics fns for the batch
                EDCorrs.update(pred_coords.numpy(), 
                               target_coords.numpy())
                PCME.update(target_coords)

        # after all batches in fold: compute, store, and reset metric-calculating classes
        fold_corrs_dict = EDCorrs.compute()
        valid_metrics_dict['scc'][i] = fold_corrs_dict['scc']
        valid_metrics_dict['pcc'][i] = fold_corrs_dict['pcc']
        EDCorrs.reset()
        
        fold_configmetrics_dict = PCME.compute()
        valid_metrics_dict['rmsd'][i] = fold_configmetrics_dict['rmsd']
        valid_metrics_dict['dope'][i] = fold_configmetrics_dict['dope']
        PCME.reset()
    else:
        valid_metrics_dict = None
        
    return dataloaders, trained_model, valid_metrics_dict


# cross-validation: train model k times, storing valid metrics (in a dict of lists)
if clargs.k_cv > 1:
    valid_metrics_dict = {metric: [None] * clargs.k_cv for metric in METRICS}
    for i in range(clargs.k_cv):
        dataloaders, trained_model, valid_metrics_dict = run_fold(i, valid_metrics_dict)
else:
    dataloaders, trained_model, valid_metrics_dict = run_fold(FOLD_TO_SAVE, None)


if CALC_METRICS:
    # compute cv results: mean and st dev
    # store as records for pandas compatibility
    cv_records = [None] * len(METRICS)
    try:
        for i, (metric, results_l) in enumerate(valid_metrics_dict.items()):
            print(f'processing {metric}...')
            cv_records[i] = {
                'metric': metric,
                'mean': np.mean(results_l),
                'st_dev': np.std(results_l),
                'cv_results': results_l
            }
    except TypeError as e:
        print('\t', e)
        pass
    except Exception as e:
        print(e)
    pass
    
    # save cv metrics results
    save_path = f'{clargs.save_results_folder}/mdfpvae_cv_records_{clargs.protein_name}.p'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(cv_records, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Results saved.')


if SAVE_LS_EMBEDDINGS:
    all_ls_embeds = []
    trained_model.eval()
    with torch.set_grad_enabled(False):
        for input_dict in dataloaders['valid']:
            batch_ls_embeds, _ = trained_model.encode(input_dict['x'])
            all_ls_embeds.extend(batch_ls_embeds)
        all_ls_embeds = torch.stack(all_ls_embeds)
        print(all_ls_embeds)
        
    embeds_save_path = f'{clargs.save_results_folder}/mdfpvae_ls_embeds_{clargs.protein_name}.p'
    os.makedirs(os.path.dirname(embeds_save_path), exist_ok=True)
    with open(embeds_save_path, "wb") as f:
        pickle.dump(all_ls_embeds, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Latent space embeddings saved.')


# print total time elapsed
t_min, t_sec = U.get_time_min_sec(time.time(), time_0)
print(f'\nDone with {clargs.k_cv}-fold cross-validation ({t_min:.0f}min, {t_sec:.1f}sec.).')


# print results
if CALC_METRICS:
    print(f'\nResults for {clargs.protein_name}:')
    for record in cv_records:
        for k, v in record.items():
            print(f'\t{k}: {v}')

"""
From here:
To open results (as pandas df):

with open(save_path, "rb") as f:
    dataset_dict = pickle.load(f)
pd.DataFrame.from_records(dataset_dict)

"""

