"""
Subclasses of `torch.utils.data.Dataset`.
"""
import torch
from torch.utils.data import Dataset


class DatasetMDFP(Dataset):
    """
    For MDFP data. 
    Note: `__getitem__` returns a dictionary
    instead of just an x/input!
    """
    def __init__(self, 
                 inputs, 
                 targets):
        super(DatasetMDFP, self).__init__()
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        targets_dict = {
            k: v[idx] \
            for k, v in self.targets.items()
        }
        data_obj_dict = {
            'x': x, 
            'target': targets_dict
        }
        return data_obj_dict