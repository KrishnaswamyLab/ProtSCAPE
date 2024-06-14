from argparse import ArgumentParser
import datetime
import os
import numpy as np

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import pickle
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
import mdtraj as md
from models.gsae_model import GSAE
from models.progsnn import ProGSNN, ProGSNN_ATLAS
from torch_geometric.loader import DataLoader
from torchvision import transforms

from deshaw_processing.de_shaw_Dataset import DEShaw, Scattering

"""
ENV
"""
import numpy as np
from numpy.random import RandomState
import numpy as np
from numpy.random import RandomState
from argparse import ArgumentParser
import datetime
import os
import numpy as np

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import pickle
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path

from models.gsae_model import GSAE
from models.progsnn import ProGSNN_ATLAS
from torch_geometric.loader import DataLoader
from torchvision import transforms
from baselines.Baseline_1.metrics.metrics_fns import calc_dope_scores
from deshaw_processing.de_shaw_Dataset import DEShaw, Scattering
from tqdm import tqdm

"""
ENV
"""

"""
Splitting seeds by protein dataset
"""
CV_SEED_GB3 = 243858
CV_SEED_BPTI = 647899
CV_SEED_UBIQ = 187349
CV_SEED_1bxy = 133538
CV_SEED_1bx7 = 988573
CV_SEED_1ptq = 781593

"""
Splitting function
"""
def get_cv_idx_l(seed, dataset_size, k):
    """
    Generates a list of valid set index arrays 
    for k folds.
    """
    rs = RandomState(seed)
    idx = np.arange(dataset_size)
    rs.shuffle(idx)
    idx_l = np.array_split(idx, k)
    return idx_l

"""
PCC function
"""
def calculate_pearson_correlation(X, Y):
    """
    Calculate the Pearson correlation coefficient separately for each dimension (X, Y, Z) across all graphs in the test set.

    Args:
        X: numpy array of shape (N, M, D), where N is the number of graphs, M is the number of residues/nodes,
           and D is the number of dimensions containing predicted coordinates.
        Y: numpy array of shape (N, M, D), where N is the number of graphs, M is the number of residues/nodes,
           and D is the number of dimensions containing ground truth coordinates.

    Returns:
        pearson_corr: numpy array of shape (D,) containing the Pearson correlation coefficient for each dimension.
    """
    N, M, D = X.shape
    pearson_corr = np.zeros(D)

    for d in range(D):
        X_d = X[:, :, d].flatten()
        Y_d = Y[:, :, d].flatten()
        pearson_corr[d] = np.corrcoef(X_d, Y_d)[0, 1]

    return pearson_corr

"""
PCC/SCC function
"""
def eucl_dist_corrs(coords_1, 
                        coords_2, 
                        squared=False):
        """
        For two sets of coordinates, calculate the Pearson correlation 
        coefficient (PCC) between their two intra-set euclidean distances, 
        optionally squared.
        """
        from sklearn.metrics.pairwise import euclidean_distances
        from scipy.stats import spearmanr
        
        dists = [None] * 2
        for i, coords in enumerate((coords_1, coords_2)):
            dist = euclidean_distances(coords, coords)
            # exclude self-distances and duplicates (i,j-th = j,i-th distance)
            # -> create lower triangular mask excluding diagonal
            tril_mask = np.tril_indices_from(dist, k=-1)
            dist = dist[tril_mask]
            if squared:
                dist = np.square(dist)
            dists[i] = dist
        # calc the 2x2 corr matrix, and return the top off-diagonal
        pcc = np.corrcoef(dists[0], dists[1])[0, 1]
        scc = spearmanr(dists[0], dists[1])
        return pcc, scc

"""
MinMax scaling function
"""
def calc_minmax_xyz_cols(train_arrays_l, verbose=False):
    """
    Calculates min and max values for each x, y, z dimension, given a list
    of (training set) arrays of shape (n, 3).
    """
    train_arrays_stack = np.row_stack(train_arrays_l)
    train_mins = np.apply_along_axis(func1d=np.min, arr=train_arrays_stack, axis=0)
    train_maxs = np.apply_along_axis(func1d=np.max, arr=train_arrays_stack, axis=0)
    if verbose:
        print('train_mins:', train_mins)
        print('train_maxs:', train_maxs)
    return (train_mins, train_maxs)


"""
MinMax scaling function pt.2
"""
def minmax_scale_arrays(data, train_mins, train_maxs, inverse_transform=False):
    """
    Min-max scales each xyz dimension for each array (of shape (n, 3))
    in `data` (if `data` is a list of arrays of this shape), given mins 
    and maxes (arrays of shape (1, 3), calculated from the list of [train set] arrays). 
    Returns a list of scaled arrays, or one rescaled array.

    This function exists so that the same mins and maxes learned on the train
    arrays can be used to scale train, valid, and test arrays.
    """
    ranges = train_maxs - train_mins
    if inverse_transform: # inverse/undo min-max scale
        scale_xyz_row = lambda row: (row * ranges) + train_mins
    else: # min-max scale
        scale_xyz_row = lambda row: (row - train_mins) / ranges
        
    if isinstance(data, list):
        scaled_arrays_l = [None] * len(data)
        for i, arr in enumerate(data):
            arr_scaled = np.apply_along_axis(func1d=scale_xyz_row, arr=arr, axis=1)
            scaled_arrays_l[i] = arr_scaled
        return scaled_arrays_l
    elif isinstance(data, np.ndarray):
        arr_scaled = np.apply_along_axis(func1d=scale_xyz_row, arr=data, axis=1)
        return arr_scaled

"""
RMSD functions
"""

def get_residue_coords(frame):
    """
    Computes a numpy array of residues' xyz-coordinates from
    an mdtraj trajectory frame.
    """
    residue_ctr_coords = [None] * frame.n_residues
    for j, residue in enumerate(frame.top.residues):
        atom_indices = [atom.index for atom in residue.atoms]
        # note that frame.xyz[0].shape = (n_atoms, 3)
        atom_coords = frame.xyz[0][atom_indices] 
        mean_coords = np.mean(atom_coords, axis=0)
        residue_ctr_coords[j] = mean_coords
    residue_ctr_coords = np.row_stack(residue_ctr_coords)
    return residue_ctr_coords


def est_atomic_pdb_from_residue_coords(orig_frame, 
                                       new_residue_coords,
                                       orig_residue_coords=None):
    """
    Generates a new mdtraj trajectory frame with all atoms within residues
    shifted by the differences between an original and new frame's center-of-
    residues' x, y, and z coordinates.

    This allows us to estimate atomic positions from new residue positions, and
    hence use functions/metrics designed for atom-level pdb files. HOWEVER, we
    aren't necessarily getting the true atomic coordinates this way: some residues
    are flexible, by definition we've coarsened to residue granularity, etc.
    """

    # calc differences in residue centers between preds and orig frame
    # (caution: relies on broadcasting (n_residue, 3)-shaped arrays)
    ctr_diff = new_residue_coords - orig_residue_coords
    # print(ctr_diff, '\nshape:', ctr_diff.shape)
    
    # shift orig frame atom coords by residue diffs
    pred_residue_ctr_coords = [None] * orig_frame.n_residues
    for j, residue in enumerate(orig_frame.top.residues):
        # print(f'residue {j}')
        atom_indices = [atom.index for atom in residue.atoms]
        # print(atom_indices)
        # note that orig_frame.xyz[0].shape = (n_atoms, 3)
        atom_coords = orig_frame.xyz[0][atom_indices]
        # print(atom_coords)
    
        shift_atom_coords = atom_coords + ctr_diff[j]
        # print(shift_atom_coords, '\n')
        pred_residue_ctr_coords[j] = shift_atom_coords
    pred_residue_ctr_coords = np.row_stack(pred_residue_ctr_coords)
    
    # make a copy of the orig frame and replace its atom coords
    new_frame = orig_frame.slice(0, copy=True)
    new_frame.xyz[0] = pred_residue_ctr_coords
    return new_frame

def get_deshaw_data_info(deshaw_folderpath):
    """
    DE Shaw pdb files are in a weird folder structure
    and filenaming convention. This function walks
    through a DE Shaw data folder and generates a
    dictionary of lists holding useful file info, all
    in the order of the sorted MD simulation timesteps.
    """
    # deshaw pdb files are grouped in subfolders
    deshaw_subfolders = sorted([
        f.path for f in os.scandir(deshaw_folderpath) \
        if f.is_dir()
    ])
    n_subf = len(deshaw_subfolders)
    records = {
        'pdb_filepaths': [],
        'suffix_vals': [],
        'timestamps': []
    }
    subf_records_l = [None] * n_subf
    
    # extract info from each pdb file, by subfolder
    for j, deshaw_subf in enumerate(deshaw_subfolders):
        subf_files = os.listdir(deshaw_subf)
        n = len(subf_files)
        subf_records = {
            'pdb_filepaths': [None] * n,
            'suffix_vals': [None] * n,
            'timestamps': [None] * n
        }
        
        for i, pdb_filename in enumerate(subf_files):
            a, b = pdb_filename.split('_')
            a, val = b.split('-')
            suffix_val = val.split('.')[0]
            # int(suffix_val) is 0-2 microseconds
            t = (int(a) * 1e4 + int(suffix_val)) / 1e4
            # print(t)
            subf_records['pdb_filepaths'][i] = f'{deshaw_subf}/{pdb_filename}'
            subf_records['suffix_vals'][i] = suffix_val
            subf_records['timestamps'][i] = t
        
        # sort subfolder info lists in timestamp order
        for k, v in subf_records.items():
            subf_records[k] = [
                x for (_, x) \
                in sorted(zip(
                    subf_records['timestamps'], 
                    subf_records[k]
                ))
            ]
        subf_records_l[j] = subf_records
    
    # create master records dict (all in sorted timestamp order)
    for k in records.keys():
        for sr in subf_records_l:
            records[k].extend(sr[k])
    return records

"""
Example usage
"""
idx_l = get_cv_idx_l(seed=CV_SEED_GB3,
                        dataset_size=9985, 
                        k=5)
# ex_data = np.arange(100)
pearson_corr_all = []
spearman_corr_all = []
rmsd_all = []
dope_all = []
for fold_i in range(5):


    parser = ArgumentParser()

    parser.add_argument('--dataset', default='deshaw', type=str)

    parser.add_argument('--input_dim', default=None, type=int)
    parser.add_argument('--latent_dim', default=64, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--embedding_dim', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)

    parser.add_argument('--alpha', default=1e-8, type=float)
    parser.add_argument('--beta', default=0.0005, type=float)
    parser.add_argument('--beta_loss', default=0.2, type=float)
    parser.add_argument('--n_epochs', default=300, type=int)
    parser.add_argument('--len_epoch', default=None)
    parser.add_argument('--probs', default=0.2)
    parser.add_argument('--nhead', default=1)
    parser.add_argument('--layers', default=1)
    parser.add_argument('--task', default='reg')
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--n_gpus', default=1, type=int)
    parser.add_argument('--save_dir', default='train_logs/', type=str)
    parser.add_argument('--residue_num', default=None, type=int)
    parser.add_argument('--protein', default=None, type=str)

    # add args from trainer
    # parser = pl.Trainer.add_argparse_args(parser)
    # parse params 
    args = parser.parse_args()

    if args.protein == 'gb3':
        deshaw_records = get_deshaw_data_info("/gpfs/gibbs/pi/krishnaswamy_smita/de_shaw/GB3")
        traj = md.load(deshaw_records['pdb_filepaths'])
        tmp_pdb_savepath = f"gb3_tmp_pdbs/"
        full_dataset = DEShaw('deshaw_processing/graphs_gb3/total_graphs.pkl')
    # full_dataset = DEShaw('deshaw_processing/graphs_gb3/total_graphs.pkl')
    if args.protein == 'bpti':
        # print("GOING INTO BPTI")
        deshaw_records = get_deshaw_data_info("/gpfs/gibbs/pi/krishnaswamy_smita/de_shaw/BPTI")
        traj = md.load(deshaw_records['pdb_filepaths'])
        tmp_pdb_savepath = f"bpti_tmp_pdbs/"
        full_dataset = DEShaw('deshaw_processing/graphs_bpti/total_graphs.pkl')
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    # print(len(full_dataset))
    # print(type(full_dataset))
    # print(full_dataset[0])
    # import pdb; pdb.set_trace()
    # train loader
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                                        shuffle=True, num_workers=15)
    # valid loader 
    valid_loader = DataLoader(val_set, batch_size=args.batch_size,
                                        shuffle=False, num_workers=15)
    full_loader = DataLoader(full_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=15)

    # logger
    now = datetime.datetime.now()
    date_suffix = now.strftime("%Y-%m-%d-%M")
    save_dir =  args.save_dir + 'progsnn_logs_run_{}_{}/'.format(args.dataset,date_suffix)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    wandb_logger = WandbLogger(name='deshaw_gb3',
                                project='progsnn', 
                                log_model=True,
                                save_dir=save_dir)
    
    wandb_logger.log_hyperparams(args)
    wandb_logger.experiment.log({"logging timestamp":date_suffix})

    # print(train_loader)
    # print([item for item in full_dataset])
    # early stopping 
    # early_stop_callback = EarlyStopping(
    #         monitor='val_loss',
    #         min_delta=0.00,
    #         patience=5,
    #         verbose=True,
    #         mode='min'
    #         )
    # print(len(val_set))
    # args.input_dim = len(train_set)
    # print()
    # import pdb; pdb.set_trace()
    args.input_dim = train_set[0].x.shape[-1]
    print(train_set[0].x.shape[-1])
    # print(full_dataset[0][0].shape)
    args.prot_graph_size = max(
            [item.edge_index.shape[1] for item in full_dataset])
    print(args.prot_graph_size)
#     import pdb; pdb.set_trace()
    args.len_epoch = len(train_loader)
    args.residue_num = full_dataset[0].x.shape[0]
    # import pdb; pdb.set_trace()
    # init module
    model = ProGSNN_ATLAS(args)

    # most basic trainer, uses good defaults
    trainer = pl.Trainer(
                        max_epochs=args.n_epochs,
                        #gpus=args.n_gpus,
                        logger = wandb_logger
                        )
    trainer.fit(model=model,
                train_dataloaders=train_loader,
                val_dataloaders=valid_loader,
                )

    ####    TESTING    ####
    coords_recon_lst = []
    with torch.no_grad():
        for x in tqdm(valid_loader):
            print("Looping through test set..")
            y_hat, z_rep, _, _, _, att_map_row,coords_recon = model(x)
            coords_recon_lst.append(coords_recon)

    coords_recon = torch.cat(coords_recon_lst, dim=0)
    coords_gt = np.array([data.coords for data in val_set])
    # import pdb; pdb.set_trace()
    tensor_list = list(coords_gt)

    # Stack the tensors along a new dimension (dimension 0)
    coords_gt = torch.stack([t for t in tensor_list], dim=0)
    dope_list = []
    for i in range(coords_recon.shape[0]):
        # grab on sample's coords from batch of samples
        # res_coords = batch_residue_coords[i, :, :].numpy()
        # timestep_id = timestep_ids[i]
        
        # ref_frame = self.ref_trajectory[timestep_id]
        ref_frame =  traj[i]
        ref_frame_residue_coords = get_residue_coords(ref_frame)
        # print(coords[i].shape)
        # print(ref_frame_residue_coords.shape)
        atomic_frame = est_atomic_pdb_from_residue_coords(
            orig_frame=ref_frame, 
            new_residue_coords=coords_recon[i].numpy(),
            orig_residue_coords=ref_frame_residue_coords
        )
        # RMSD
        # rmsd = md.rmsd(target=atomic_frame, reference=ref_frame)
        # print(rmsd)
        # atomic_frame.save_pdb(f"1ptq_gt_pdb/gt_atomic_frame_{i}.pdb")
        # rmsd_list.append(rmsd)
        #DOPE
        dope = calc_dope_scores(atomic_frame.xyz[0], ref_frame, tmp_pdb_savepath=tmp_pdb_savepath, normalize=True, verbosity=1)
        dope_list.append(dope)
    
    dope_all.append(np.mean(dope_list))

print("Mean DOPE:", np.mean(dope_all))
print("Standard Deviation of DOPE:", np.std(dope_all))
"""
RMSD calculation
"""
#     rmsd_list = []
#     for i in range(coords_recon.shape[0]):
#         # grab on sample's coords from batch of samples
#         # res_coords = batch_residue_coords[i, :, :].numpy()
#         # timestep_id = timestep_ids[i]
        
#         # ref_frame = self.ref_trajectory[timestep_id]
#         ref_frame =  traj[i]
#         ref_frame_residue_coords = get_residue_coords(ref_frame)
#         # print(coords[i].shape)
#         # print(ref_frame_residue_coords.shape)
#         atomic_frame = est_atomic_pdb_from_residue_coords(
#             orig_frame=ref_frame, 
#             new_residue_coords=coords_recon[i].numpy(),
#             orig_residue_coords=ref_frame_residue_coords
#         )
#         # RMSD
#         rmsd = md.rmsd(target=atomic_frame, reference=ref_frame)
#         # print(rmsd)
#         rmsd_list.append(rmsd)
    
#     rmsd_all.append(np.mean(rmsd_list))

# print("Mean RMSD:", np.mean(rmsd_all))
# print("Standard Deviation of RMSD:", np.std(rmsd_all))

"""
PCC/SCC calculation
"""
#     pcc_lst = []
#     scc_lst = []
#     for i in range(coords_recon.shape[0]):
#         pcc, scc = eucl_dist_corrs(coords_recon[i], coords_gt[i], squared=False)
#         pcc_lst.append(pcc)
#         scc_lst.append(scc.correlation)
#     # import pdb; pdb.set_trace()

#     # pearson_corr = calculate_pearson_correlation(coords_recon, coords_gt)
#     # pearson_corr_all.append(pearson_corr)
#     pearson_corr_all.append(np.mean(pcc_lst))
#     spearman_corr_all.append(np.mean(scc_lst))
#     # print("Pearson Correlation Coefficient:", pearson_corr)

# pearson_corr_all = np.array(pearson_corr_all)
# mean_pcc = np.mean(pearson_corr_all, axis=0)
# std_dev_pcc = np.std(pearson_corr_all, axis=0)

# spearman_corr_all = np.array(spearman_corr_all)
# mean_scc = np.mean(spearman_corr_all, axis=0)
# std_dev_scc = np.std(spearman_corr_all, axis=0)

# print("Mean Pearson Correlation Coefficients:", mean_pcc)
# print("Standard Deviation of Pearson Correlation Coefficients:", std_dev_pcc)

# print("Mean Spearman Correlation Coefficients:", mean_scc)
# print("Standard Deviation of Spearman Correlation Coefficients:", std_dev_scc)

    
