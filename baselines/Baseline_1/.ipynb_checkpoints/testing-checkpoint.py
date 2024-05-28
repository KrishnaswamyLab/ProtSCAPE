"""

"""
import VAE
import VAE_latent_space as VAELS
from torchmetrics.functional.pairwise import pairwise_euclidean_distance
import numpy as np
import torch
from torchmetrics import MetricCollection
from torchmetrics.regression import (
    PearsonCorrCoef,
    SpearmanCorrCoef
)

# def calc_eucl_dist_corrs(input_data_matrix, 
#                          ls_data_matrix, 
#                          train_mins,
#                          train_maxs,
#                          orig_scale=True):
#     """
#     Calculates pearson and spearman correlation
#     coefficients between (optionally scaled or 
#     unscaled) input pts and the same pts in the latent 
#     space.

#     Returns only the correlation coefficients, though
#     the scipy objects also contain p-values.
#     """
#     from scipy.stats import spearmanr, pearsonr

#     if orig_scale:
#         input_matrix_final = DU.minmax_scale_arrays(
#             data=input_data_matrix,
#             train_mins=train_mins,
#             train_maxs=train_maxs,
#             inverse_transform=True
#         )
#     else:
#         input_matrix_final = input_data_matrix
        
#     dists_sq_l = [None] * 2
#     for i, m in enumerate([input_matrix_final, ls_data_matrix]):
#         dists = euclidean_distances(m, m) 
#         print(dists)
#         dists_sq_l[i] = np.square(dists).flatten()
        
#     corrs_dict = {}
#     corrs_dict['pcc'] = pearsonr(dists_sq_l[0], dists_sq_l[1])[0]
#     corrs_dict['scc'] = spearmanr(dists_sq_l[0], dists_sq_l[1])[0]
#     return corrs_dict
    

def calc_eucl_dist_corrs(input_dataloader, 
                         encodings_dataloader):
    """
    Calculates Pearson and Spearman correlation
    coefficients of euclidean distances between 
    the (scaled) input coordinates and the euclidean 
    distances between those points in the latent space.
    
    Note we used SCALED inputs, because the latent space 
    couldn't learn the original arbitrary scale it didn't see!

    BUT: Is correlation between spaced of two different dimensions possible?
    If you can preserve pair-wise distances...?
    https://stats.stackexchange.com/questions/351212/do-autoencoders-preserve-distances

    One correlation coefficient is calculated per sample,
    and then the specified average is taken.

    TODO
    [ ] have to match sample ids between inputs and encodings?
    [ ] metric_collection update step
    """
    metric_collection = MetricCollection({
        'pcc': PearsonCorrCoef(),
        'scc': SpearmanCorrCoef()
    })

    dists_dict = {}
    for key, set_dataloader in {'orig': input_dataloader, 
                                'encoded': encodings_dataloader}:
        dists_dict[key] = {'sample_ids': [], }
        # orig shape: [batch_size, n_coords_per_sample, 3]
        # encoded shape: [batch_size, 1, 4]
        for batch in set_dataloader: 
            pairwise_eucl_dist_matrix = pairwise_euclidean_distance(batch['x'])
            n_dists = pairwise_eucl_dist_matrix.shape[0]
            pairwise_eucl_dist_vect = pairwise_eucl_dist_matrix[np.triu_indices(n_dists, 1)]
            dists_dict[key].append(pairwise_eucl_dist_vect)

    # update and compute metric_collection, and return dict of its metrics as floats
    metrics = metric_collection.compute()
    metric_collection.reset()
    corrs_dict = { k: m.detach().cpu().numpy().item() for (k, m) in metrics.items() }
    return corrs_dict


def calc_latent_space_metrics(vae, 
                              dataloaders, 
                              # data_transform_info, 
                              set='test',
                              verbose=True):
    """

    """
    # test set batches should be shape [batch_size, 1, 3]
    encodings_dataloaders = VAE.get_VAE_encodings_by_set(vae, 
                                                         dataloaders,
                                                         sets=(set))
        
    corrs_dict = calc_eucl_dist_corrs(
        input_dataloader=dataloaders[set],
        encodings_dataloader=encodings_dataloaders[set])
        # train_mins=data_transform_info['train_mins'], 
        # train_maxs=data_transform_info['train_maxs'],
        # orig_scale=False
    )
    
    if verbose: 
        for k, v in corrs_dict.items():
            print(f'{k}: {v:.4f}')

