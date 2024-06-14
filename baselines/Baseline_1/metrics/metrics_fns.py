"""
Functions to fit and plot training metrics.
"""

import os
import sys
sys.path.insert(0, '../')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from data_processing.data_utilities import (
#     minmax_scale_arrays
# )
# print("Current working directory:", os.getcwd())
# print(sys.path)
# for DOPE
# from metrics import dope_score as ds
from baselines.Baseline_1.metrics import dope_score as ds
import mdtraj as md
import biobox
import pickle


def convert_batch_coords(batch_tensor, start_i=0):
    """
    Converts `batch_tensor` of (e.g.) shape (batch_size, 1 + n*3) to a
    (batch_size, n, 3) tensor.
    - must have xs stacked on ys on zs in input and target tensors
    - exclude (e.g. timestep) stacked on top of input vector with `start_i`
    """
    import torch
    split_size = int((batch_tensor.shape[1] - start_i) / 3)
    unstacked_coords = torch.stack(
        batch_tensor[:, start_i:].split(split_size, dim=1),
        dim=2
    )
    return unstacked_coords


def get_residue_coords(frame):
    """
    Computes a numpy array of residues' xyz-coordinates from
    an mdtraj trajectory frame.
    """
    residue_ctr_coords = [None] * frame.n_residues
    for j, residue in enumerate(frame.top.residues):
        atom_indices = [atom.index for atom in residue.atoms]
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
    if orig_residue_coords is None:
        # get the original frame's center-of-residue coords if not pre-calc'd
        orig_residue_coords = get_residue_coords(orig_frame)
    
    # calc differences in residue centers between preds and orig frame
    # (caution: relies on broadcasting (n_residue, 3)-shaped arrays)
    ctr_diff = new_residue_coords - orig_residue_coords
    # print(ctr_diff, '\nshape:', ctr_diff.shape)
    
    # shift orig frame atom coords by residue diffs
    pred_residue_ctr_coords = [None] * orig_frame.n_residues
    for j, residue in enumerate(orig_frame.top.residues):
        atom_indices = [atom.index for atom in residue.atoms]
        atom_coords = orig_frame.xyz[0][atom_indices]
    
        shift_atom_coords = atom_coords + ctr_diff[j]
        pred_residue_ctr_coords[j] = shift_atom_coords
    pred_residue_ctr_coords = np.row_stack(pred_residue_ctr_coords)
    
    # make a copy of the orig frame and replace its atom coords
    new_frame = orig_frame.slice(0, copy=True)
    new_frame.xyz[0] = pred_residue_ctr_coords
    return new_frame
    

def calc_dope_scores(coords_arr,
                     ref_frame,
                     tmp_pdb_savepath,
                     normalize = False,
                     refine = False,
                     multiproc = False, 
                     multiproc_kwargs = {'n_proc': 4, 
                                         'context': 'spawn'},
                     addl_mp_pool_kwargs = {},
                     verbosity = 0):
    """
    Arg `coords_arr` can be shape (batch_size, n_atoms, 3) or (n_atoms, 3); 
    will return list of dope scores for batch, or single dope score
    accordingly.

    TODO
    [ ] parallelizing broken; requires something like if __name__ == '__main__':
    wrapping as seen in `example_dope.py`
    """
    # print("Random")
    # create a biobox.Molecule object from pdb
    # this should have 1 frame; only used to get topology
    # for both ATLAS and DE Shaw, we can grab the first frame 
    # from the reference trajectory, and save as a temporary pdb
    mol = biobox.Molecule()
    path = f"{tmp_pdb_savepath}/tmp_frame.pdb"
    # os.makedirs(path, exist_ok=True)
    ref_frame.save_pdb(path, force_overwrite=True)
    mol.import_pdb(path)
    # print(len(coords_arr.shape))
    # if a batch of frames was passed to calc multiple DOPEs:
    if len(coords_arr.shape) == 3: # (b, n_atoms, 3)
        if multiproc:
            # print("Going option 1")
            """
            Parallelized DOPE scores
            """
            # empty list for dope scores
            dopes = [None] * coords_arr.shape[0]
        
            # mp_pool_kwargs passed to `multiprocesing.pool` during init
            parallel_dope_score_obj = ds.Parallel_DOPE_Score(
                mol=mol, 
                normalize=normalize,
                processes=multiproc_kwargs['n_proc'],
                context=multiproc_kwargs['context'],
                **addl_mp_pool_kwargs
            )
            # DOPE is calculated asynchronously in background
            for i, coords in enumerate(coords_arr):
              dopes[i] = parallel_dope_score_obj.get_score(coords)
            
            # retrieve the results after multiprocess finishes
            dope_scores = np.array([d.get() for d in dopes])
        
        else:
            """
            Serial DOPE scores
            """
            # print("Going option 2") 
            # init dope score calculator object
            dope_score_obj = ds.DOPE_Score(mol, normalize)

            # option 1: loop
            # note any error in MODELLER just returns 1e10
            # for i, coords in enumerate(coords_l):
            #     dopes[i] = dope_score_obj.get_dope(coords)
            # dope_scores = dopes
            
            # option 2: `DOPE_Score`'s optimized serial process
            dope_scores = dope_score_obj.get_all_dope(coords_arr, refine=refine)
    
        if verbosity > 0:
            print('dope results:')
            for s in dope_scores:
                print(f'\t{s:.4f}')
        return dope_scores

    # if only 1 frame was passed (shape (n, 3)), for 1 DOPE score:
    elif len(coords_arr.shape) == 2:
        # print("Going option 3")
        dope_score_obj = ds.DOPE_Score(mol, normalize)
        dope = dope_score_obj.get_dope(coords_arr, refine=refine)
        if verbosity > 0:
            # print("here")
            print(f'dope = {dope:.4f}')
        return dope


class ProteinConfigMetricsEstimates:
    """
    Computes protein configuration metrics between a reference trajectory
    frame and another frame, back-estimated at the atomic level from
    residue-level coordinates. Metrics: 
    - root-mean squared deviation (RMSD)
    - solvent-accessible surface area (SASA)
    - radius of gyration (RG)
    - discrete-optimized protein energy (DOPE)
    
    Works with a torch.nn module by updating in batch tensors of shape 
    (batch_size, n, 3) [might have to use `convert_batch_coords` function 
    from above in this file, upstream].
    """
    
    def __init__(self, 
                 ref_frame,
                 transform_info=None, # dict of data scaling info (e.g. axis min/maxs)
                 reduction='mean',
                 dope_kwargs={},
                 verbosity = 0):
        """
        NOTE: Use the first frame as a template to back-calculate atomic coords
        from predicted residue coords. (If you use the ground truth frame 
        from the same timestep to back-calculate atomic coords from pred.
        residue coords, you're adding atomic info the model didn't learn,
        and biasing the predicted atomic coords to be more correct.)
        """
        self.ref_frame = ref_frame
        self.transform_info = transform_info
        self.reduction = reduction
        self.dope_kwargs = dope_kwargs
        self.verbosity = verbosity
        self.rmsds = []
        self.dopes = []

    def update(self, batch_pred_residue_coords):
        
        batch_size = batch_pred_residue_coords.shape[0]
        batch_rmsds = [None] * batch_size
        batch_atomic_coords = [None] * batch_size
        
        for i in range(batch_size):
            # grab on sample's coords from batch of samples
            res_coords = batch_pred_residue_coords[i, :, :].numpy()
            
            # unscale residue coords back to original scales
            # if self.transform_info is not None:
            #     res_coords = minmax_scale_arrays(
            #         data=res_coords, 
            #         train_mins=self.transform_info['train_mins'], 
            #         train_maxs=self.transform_info['train_maxs'], 
            #         inverse_transform=True
            #     )
                
            # back-estimate atomic coords from pred. residue coords
            # !!! use the single ref_frame (see note in fn annotation)
            ref_frame_residue_coords = get_residue_coords(self.ref_frame)
            atomic_frame = est_atomic_pdb_from_residue_coords(
                orig_frame=self.ref_frame, 
                new_residue_coords=res_coords,
                orig_residue_coords=ref_frame_residue_coords
            )
            # RMSD
            rmsd = md.rmsd(target=atomic_frame, reference=self.ref_frame)
            batch_rmsds[i] = rmsd.item()

            # DOPE
            # collect atomic coords in a list for batch calculation
            # (one by one is very slow!)
            batch_atomic_coords[i] = atomic_frame.xyz[0]

        # after batch processing is complete
        self.rmsds.extend(batch_rmsds)

        # Batch DOPEs
        batch_dopes = calc_dope_scores(
            coords_arr=np.stack(batch_atomic_coords, axis=0),
            ref_frame=self.ref_frame,
            **self.dope_kwargs
        )
        self.dopes.extend(batch_dopes)

    def compute(self):
        results_dict = {}
        if self.reduction == 'mean':
            results_dict['rmsd'] = np.mean(self.rmsds)
            results_dict['dope'] = np.mean(self.dopes)
        return results_dict

    def reset(self):
        self.rmsds = []
        self.dopes = []
        

class EuclideanDistanceCorrs:
    """
    Pearson and Spearman correlation coefficients for within-configuration
    euclidean distances between residues (optionally squared), between two 
    configurations of the same protein. Works with a torch.nn module by
    updating in batch tensors of shape (batch_size, n, 3) [might have to 
    use `convert_batch_coords` function above, upstream].
    """    
    def __init__(self, 
                 square_dists=False,
                 reduction='mean'):
        self.square_dists = square_dists
        self.reduction = reduction
        self.pccs = []
        self.sccs = []
    
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
        # if an error is encountered in corr calc, return 0.0, not NaN
        try:
            # calc the 2x2 corr matrix, and return the top off-diagonal
            pcc = np.corrcoef(dists[0], dists[1])[0, 1]
        except:
            pcc = 0.0
        try:
            # `spearmanr` returns a `SignificanceResult` object
            # coerr. coeff. is at index 0
            scc = spearmanr(dists[0], dists[1])[0]
        except:
            scc = 0.0
        return pcc, scc

    def update(self,
               batch_coords_1, 
               batch_coords_2):
        batch_size = batch_coords_1.shape[0]
        batch_pccs, batch_sccs = [None] * batch_size, [None] * batch_size
        for i in range(batch_size):
            pcc, scc = EuclideanDistanceCorrs.eucl_dist_corrs(
                coords_1=batch_coords_1[i, :, :],
                coords_2=batch_coords_2[i, :, :],
                squared=self.square_dists
            )
            batch_pccs[i] = pcc
            batch_sccs[i] = scc
        
        cleaned_batch_pccs = [c for c in batch_pccs if not np.isnan(c)]
        cleaned_batch_sccs = [c for c in batch_sccs if not np.isnan(c)]
        self.pccs.extend(cleaned_batch_pccs)
        self.sccs.extend(cleaned_batch_sccs)

    def compute(self):
        results_dict = {}
        if self.reduction == 'mean':
            results_dict['pcc'] = np.mean(self.pccs).item()
            results_dict['scc'] = np.mean(self.sccs)
        return results_dict

    def reset(self):
        self.pccs = []
        self.sccs = []
    

def show_train_plot(history_df,
                    metrics_l, 
                    burnin_n_epochs=0,
                    smooth=False,
                    smooth_window=10,
                    vline_x=None,
                    hline_y=None,
                    title=None,
                    legend_loc='upper right',
                    fig_size=(10, 5),
                    y_lim=(-0.05, 1.05),
                    y_scale_log=False,
                    grid_step_x='auto'):
    """
    Constructs a (multi-)line plot for the metrics in 
    `history_df`, showing their values by training epoch.
    """
    # figure setup
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams['grid.color'] = 'lightgray'

    # x-axis
    if grid_step_x == 'auto':
        grid_step_x_used = len(history_df) // 16
    else:
        grid_step_x_used = grid_step_x
    grid_step_x_used = max([2, grid_step_x_used])
    x_ticks = range(burnin_n_epochs + 1, 
                    history_df.epoch.max() + 1, 
                    grid_step_x_used)
        
    if smooth:
        alpha_orig_data = 0.2
        metrics_smooth = [None] * len(metrics_l)
        for i, metric in enumerate(metrics_l):
            metric_data = history_df[metric]
            
            # convolution (each mode has diff. edge artifacts)
            smooth_valid = np.convolve(
                metric_data, 
                np.ones(smooth_window) / smooth_window, 
                mode='valid'
            )
            len_diff = len(metric_data) - len(smooth_valid)
            smooth_padded = np.concatenate([
                smooth_valid,
                np.full((len_diff, ), smooth_valid[-1])
            ])
            metrics_smooth[i] = smooth_padded
    else:
        alpha_orig_data = 1.0

    fig, ax1 = plt.subplots()

    # metric 1 plotting
    color1 = 'tab:blue'
    ax1.set_xlabel('epoch')
    ax1.set_ylabel(metrics_l[0], 
                   color=color1)
    ax1.plot(history_df['epoch'][burnin_n_epochs:], 
             history_df[metrics_l[0]][burnin_n_epochs:], 
             color=color1,
             alpha=alpha_orig_data)
    ax1.tick_params(axis='y', 
                    labelcolor=color1)
    ax1.set_ylim(y_lim)
    if y_scale_log:
        ax1.set_yscale('log')
    if smooth:
        ax1.plot(history_df['epoch'], 
                 metrics_smooth[0], 
                 color=color1)

    # metric 2 plotting
    # if there are 2 metrics, give them separate y axes
    if len(metrics_l) == 2:
        # metric 2
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel(metrics_l[1], 
                       color=color2)
        ax2.plot(history_df['epoch'][burnin_n_epochs:], 
                 history_df[metrics_l[1]][burnin_n_epochs:], 
                 color=color2,
                 alpha=alpha_orig_data)
        ax2.tick_params(axis='y', 
                        labelcolor=color2)
        ax2.set_ylim(y_lim)
        if y_scale_log:
            ax2.set_yscale('log')
        if smooth:
            ax2.plot(history_df['epoch'], 
                     metrics_smooth[1], 
                     color=color2)               
            
    # hline and vline options
    if vline_x is not None:
        plt.axvline(x=vline_x, 
                    color='gray', 
                    ls='--')
    if hline_y is not None:
        plt.axhline(y=hline_y, 
                    color='gray', 
                    ls='--')

    # legend and title options
    if legend_loc is not None:
        plt.legend(framealpha=1.0, 
                   loc=legend_loc)
    if title is not None:
        plt.title(title)

    plt.xticks(x_ticks)
    plt.grid()
    plt.show()

