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
import mdtraj as md
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

"""
Example usage
"""
idx_l = get_cv_idx_l(seed=CV_SEED_GB3,
                        dataset_size=1001, 
                        k=5)
# ex_data = np.arange(100)
pearson_corr_all = []
spearman_corr_all = []
rmsd_all = []
dope_all = []
for fold_i in range(5):

    # Training ProGSNN_atlas

    parser = ArgumentParser()

    parser.add_argument('--dataset', default='atlas', type=str)

    parser.add_argument('--input_dim', default=None, type=int)
    parser.add_argument('--latent_dim', default=64, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--embedding_dim', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)

    parser.add_argument('--alpha', default=1e-8, type=float)
    parser.add_argument('--beta', default=0.0005, type=float)
    parser.add_argument('--beta_loss', default=0.2, type=float)
    parser.add_argument('--gamma', default=0.0005, type=float)
    parser.add_argument('--n_epochs', default=300, type=int)
    parser.add_argument('--len_epoch', default=None)
    parser.add_argument('--probs', default=0.2)
    parser.add_argument('--nhead', default=1)
    parser.add_argument('--layers', default=1)
    parser.add_argument('--task', default='reg')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--n_gpus', default=1, type=int)
    parser.add_argument('--save_dir', default='train_logs/', type=str)
    parser.add_argument('--residue_num', default=None, type=int)
    parser.add_argument('--protein', default=None, type=str)
    # add args from trainer
    # parser = pl.Trainer.add_argparse_args(parser)
    # parse params 
    args = parser.parse_args()

    #55 residues
    if args.protein == '1bx7':
        #Change to analyis of 1bgf_A_protein
        traj = md.load("1bx7_A_analysis/1bx7_A_R1.xtc", top= "1bx7_A_analysis/1bx7_A.pdb")
        tmp_pdb_savepath=f"1bx7_tmp_pdb/"
        with open('1bx7_A_analysis/graphsrog.pkl', 'rb') as file:
            full_dataset =  pickle.load(file)
    #46 residues
    if args.protein == '1ab1':
        with open('1ab1_A_analysis(1)/graphsrog.pkl', 'rb') as file:
            full_dataset =  pickle.load(file)
    #60 residues
    if args.protein == '1bxy':
        traj = md.load("1bxy_A_analysis/1bxy_A_R1.xtc", top= "1bxy_A_analysis/1bxy_A.pdb")
        tmp_pdb_savepath=f"1bxy_tmp_pdb/"
        with open('1bxy_A_analysis/graphsrog.pkl', 'rb') as file:
            full_dataset =  pickle.load(file)
            
    if args.protein == '1ptq':
        traj = md.load("1ptq_A_analysis/1ptq_A_R1.xtc", top= "1ptq_A_analysis/1ptq_A.pdb")
        tmp_pdb_savepath=f"1ptq_tmp_pdbs/"
        with open('1ptq_A_analysis/graphsrog.pkl', 'rb') as file:
            full_dataset =  pickle.load(file)
    if args.protein == '1fd3':
        with open('1fd3_A_analysis/graphsrog.pkl', 'rb') as file:
            full_dataset =  pickle.load(file)
    
    
    idx = idx_l[fold_i]
    train_mask = np.full(len(full_dataset), True, dtype=bool)
    train_mask[idx] = False
    train_set = [full_dataset[i] for i in range(len(full_dataset)) if train_mask[i]]
    val_set = [full_dataset[i] for i in idx]
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
    
    wandb_logger = WandbLogger(name=f'atlas_{args.protein}',
                                project='progsnn', 
                                log_model=True,
                                save_dir=save_dir)
    
    wandb_logger.log_hyperparams(args)
    wandb_logger.experiment.log({"logging timestamp":date_suffix})

    args.input_dim = train_set[0].x.shape[-1]
    print(train_set[0].x.shape[-1])
    # print(full_dataset[0][0].shape)
    args.prot_graph_size = max(
            [item.edge_index.shape[1] for item in full_dataset])
    print(args.prot_graph_size)

    args.len_epoch = len(train_loader)
    #Set number of residues args here
    args.residue_num = full_dataset[0].x.shape[0]
    # init module
    model = ProGSNN_ATLAS(args)

    # most basic trainer, uses good defaults
    trainer = pl.Trainer(
                        max_epochs=args.n_epochs,
                        devices = "auto",
                        #gpus=args.n_gpus,
                        #callbacks=[early_stop_callback],
                        logger = wandb_logger
                        )
    trainer.fit(model=model,
                train_dataloaders=train_loader,
                val_dataloaders=valid_loader,
                )

    
    # model = model.cpu()
    # model.dev_type = 'cpu'
    # print('saving model')
    # torch.save(model.state_dict(), save_dir + f"model_atlas_{args.protein}.npy")

    coords_recon_lst = []
    with torch.no_grad():
        for x in tqdm(valid_loader):
            print("Looping through test set..")
            y_hat, z_rep, _, _, _, att_map_row,coords_recon = model(x)
            coords_recon_lst.append(coords_recon)
    
    coords_recon = torch.cat(coords_recon_lst, dim=0)
    coords_gt = np.array([data.coords for data in val_set])

        # import mdtraj on first call


# batch_rmsds = [None] * batch_size
# batch_sasas = [None] * batch_size
# batch_rgs = [None] * batch_size
# batch_dopes = [None] * batch_size
# deshaw_records = get_deshaw_data_info("/gpfs/gibbs/pi/krishnaswamy_smita/de_shaw/BPTI")

# traj = md.load(deshaw_records['pdb_filepaths'])
    # traj = md.load('1ptq_A_analysis/1ptq_A_R1.xtc', top='1ptq_A_analysis/1ptq_A.pdb')
    # rmsd_list = []
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
Calculate RMSD
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
Calculate PCC and SCC
"""
#     pcc_lst = []
#     scc_lst = []
#     for i in range(coords_recon.shape[0]):
#         pcc, scc = eucl_dist_corrs(coords_recon[i], coords_gt[i], squared=False)

#         pcc_lst.append(pcc)
#         scc_lst.append(scc.correlation)
#     # pearson_corr = calculate_pearson_correlation(coords_recon, coords_gt)
#     # pearson_corr_all.append(pearson_corr)
#     pearson_corr_all.append(np.mean(pcc_lst))
#     spearman_corr_all.append(np.mean(scc_lst))

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


    



    
            

    


