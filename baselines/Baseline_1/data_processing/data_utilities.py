"""
Utility functions for data processing.
"""
import numpy as np
from numpy.random import RandomState
import random


def get_cv_idx_l(seed, dataset_size, k=5):
    """
    Generates a list index arrays for k folds.
    E.g., if k = 5, get 80/20 test/valid split.
    """
    rs = RandomState(seed)
    idx = np.arange(dataset_size)
    rs.shuffle(idx)
    idx_l = np.array_split(idx, k)
    return idx_l


def get_train_valid_sets(dataset, valid_idx):
    """
    Given a dataset and list of valid set idx, returns
    train and valid datasets.
    """
    from torch.utils.data.dataset import Subset
    dataset_size = len(dataset)
    all_idx = np.arange(dataset_size)
    train_mask = np.full(dataset_size, True, dtype=bool)
    train_mask[valid_idx] = False
    train_idx = all_idx[train_mask]
    train_data, valid_data = Subset(dataset, train_idx), Subset(dataset, valid_idx)
    return train_data, valid_data


def get_train_splits_idx_dict(seed, sample_size, train_prop, valid_prop):
    """
    Creates a dictionary where key = set [train/valid/test], and
    value = list of indexes into the full dataset for that set.
    """
    random.seed(seed)
    shuffled_idx = random.sample(range(sample_size), k=sample_size) 
    train_valid_cut_i = int(train_prop * sample_size)
    valid_test_cut_i = train_valid_cut_i + int(valid_prop * sample_size)
    set_idx_dict = {
        'train': shuffled_idx[:train_valid_cut_i],
        'valid': shuffled_idx[train_valid_cut_i:valid_test_cut_i],
        'test': shuffled_idx[valid_test_cut_i:]
    }
    return set_idx_dict


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


def minmax_scale_arrays(data, 
                        train_mins, 
                        train_maxs, 
                        inverse_transform=False):
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
            arr_scaled = np.apply_along_axis(
                func1d=scale_xyz_row, 
                arr=arr, 
                axis=1
            )
            scaled_arrays_l[i] = arr_scaled
        return scaled_arrays_l
    elif isinstance(data, np.ndarray):
        arr_scaled = np.apply_along_axis(
            func1d=scale_xyz_row, 
            arr=data, 
            axis=1
        )
        return arr_scaled

