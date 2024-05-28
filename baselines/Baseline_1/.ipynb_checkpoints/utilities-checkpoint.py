"""
UTILITY FUNCTIONS AND CLASSES
"""


"""
``MLArgumentParser`` class
----------------------
Extends argparse.ArgumentParser to add
default args on init
"""
import argparse
class MLArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super(MLArgumentParser, self).__init__()
        self.add_argument('-a', '--machine', default='local', type=str,
                    help='machine name, used to set correct directories (default: local)')
        self.add_argument('-n', '--model_name', default=None, type=str,
                          help='model name')
        self.add_argument('-f', '--save_final', default='t', type=str,
                          help='save final training epoch model state (default: t/True)')
        self.add_argument('-m', '--min_epochs', default='5', type=int,
                          help='min num of train epochs (default: 5)')
        self.add_argument('-x', '--max_epochs', default='25', type=int,
                          help='max num of train epochs (default: 25)')
        self.add_argument('-b', '--batch_size', default='64', type=int,
                          help='batch size (default: 64)')
        self.add_argument('-l', '--learn_rate', default='0.001', type=float,
                          help='optimizer learning rate (default: 0.001)')
        self.add_argument('-s', '--snapshot_name', default='None', type=str,
                          help='name of model snapshot to restart training from')
        self.add_argument('-v', '--verbose', default='False', type=bool,
                          help='controls console printing during training')

"""
``parent_path()``
-----------------
Get parent directory path for a filepath
"""
def parent_path(path):
    from pathlib import Path
    return Path(path).parent.absolute()


"""
``print_and_log()``
Prints text to the console and
writes it to a log file.
-----------------
"""
def print_and_log(out, logfile_path):
    print(out)
    with open(logfile_path, 'a') as f:
        f.write(out + "\n")


"""
``pickle_obj()``
-----------------
More robust pickling function
"""
def pickle_obj(path, obj):
    import pickle
    
    if (path is not None) and (path != ""):
        try:
            with open(path, "wb") as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        except pickle.UnpicklingError as e:
            # normal, somewhat expected
            pass
        except (AttributeError,  EOFError, ImportError, IndexError) as e:
            # secondary errors
            print(e)
        except FileNotFoundError as e:
            print(e)
            print(f"FileNotFoundError: attempted path: {path}")
            print("File not saved!")
            return
        except Exception as e:
            # everything else, possibly fatal
            print(e)
            print(f"Attempted path: {path}")
            print("File not saved!")
            return
    else:
        print("No save path given; file not saved!")


"""
``get_time_min_sec()``
----------------------
Calculates whole min and fractional seconds
between two time points.
"""
def get_time_min_sec(t_1, t_0):
    t = t_1 - t_0
    t_min, t_sec = t // 60, t % 60
    return t_min, t_sec


"""
``build_ffnn()``
----------------
Builds a simple feed-forward, fully-connected neural 
network (aka multilayer perceptron, or MLP) programatically.

Note: returns pieces that model's `forward()` must iterate
through, e.g.
def forward(self, x):
    for i in range(len(self.lin_fns)):
        x = self.lin_fns[i](x)
        x = self.nonlin_fns[i](x)
        if self.use_dropout:
            x = nn.Dropout(self.dropout_p)
    x = self.lin_out(x)
"""
def build_ffnn(input_dim, 
               output_dim, 
               hidden_dim_array, 
               nonlin_fn,
               nonlin_fn_kwargs):
    import torch.nn as nn
    lin_fns = [None] * len(hidden_dim_array)
    next_input_dim = input_dim
    for i, hidden_dim in enumerate(hidden_dim_array):
        lin_fns[i] = nn.Linear(next_input_dim, hidden_dim)
        next_input_dim = hidden_dim
    lin_fns = nn.ModuleList(lin_fns)
    nonlin_fns = [nonlin_fn(**nonlin_fn_kwargs)] * len(hidden_dim_array)
    final_lin = nn.Linear(next_input_dim, output_dim)
    return (lin_fns, nonlin_fns, final_lin)


"""
``pad_arrays_rowwise()``
------------------------
Pads all arrays in a list to the same number of rows, 
extending shorter arrays by filling in with the 
`pad_value`.
"""

def pad_arrays_rowwise(arrays_l, max_nrow, pad_value):
    import numpy as np
    pad_len_l = [None] * len(arrays_l)
    padded_arrays = [None] * len(arrays_l)
    for i, arr in enumerate(arrays_l):
        n_rows = arr.shape[0]
        n_pads = max_nrow - n_rows
        pad_arr = np.full((n_pads, 3), pad_value)
        pad_len_l[i] = n_pads
        padded_arrays[i] = np.row_stack((arr, pad_arr))
    return (padded_arrays, pad_len_l)


"""
``get_datasets()``
------------------
Takes an original dataset (`orig_data`), plus a dataset 
processing function (`proc_data_fn`) and a subclass of torch's
`Dataset` class (`dataset_class`), along with a `set_idx_dict`,
for example = {'train': [0, 1, 3, 5],
               'valid': None,
               'test': [2, 4]},
and a list of labels; returns a dictionary of datasets of the 
`dataset_class` type by set.

Notes:
- the `proc_data_fn` must take (1) the original data and (2)
a dictionary of 'train' and 'valid' index sets (`set_idx_dict`)
indexing into the original data as its first 2 args. 
- the `proc_data_fn` used in this fn must return a dict of 
  'set': {} # e.g. 'train', 'valid'
      'inputs': []
      'targets': [] # e.g. labels for classification

Additional args can be passed in `proc_data_fn_kwargs`.

"""
def get_datasets(set_idx_dict, 
                 # labels, # moved into `proc_data_fn_kwargs`...
                 orig_data,
                 proc_data_fn,
                 proc_data_fn_kwargs,
                 dataset_class,
                 dataset_class_kwargs):
    # filter out any set keys (e.g. 'valid') storing 'None' values
    set_idx_dict_filt = {k:v for (k, v) in set_idx_dict.items() if v is not None}
    proc_data_dict = proc_data_fn(orig_data, 
                                  set_idx_dict_filt, 
                                  **proc_data_fn_kwargs)
    
    datasets = {}
    for set, dict in proc_data_dict.items():
        datasets[set] = dataset_class(inputs=dict['inputs'], 
                                      targets=dict['targets'],
                                      **dataset_class_kwargs)
    return datasets


"""
``get_dataloaders()``
---------------------
Loads output of ``get_datasets()`` [dict of dataset
objects under 'train'/'valid'/'test' keys] into a similar
dictionary of torch DataLoaders (or subclass thereof,
if `dataloader_class` arg is not None).

References:
DataLoader: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
pin_memory: https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723
"""
def get_dataloaders(datasets, # dictionary
                    seed,
                    batch_size, 
                    dataloader_class=None,
                    num_workers=0,
                    drop_last=False,
                    pin_memory=False):
    if dataloader_class is None:
        from torch.utils.data import DataLoader 
        from torch import Generator
        dataloader_class = DataLoader
        generator = Generator()
        generator.manual_seed(seed)
        dataloader_kwargs = {'num_workers': num_workers,
                             'pin_memory': pin_memory,
                             'drop_last': drop_last,
                             'generator': generator}
    dataloaders = {}
    for set_name, dataset in datasets.items():
        shuffle = ('train' in set_name)
        dataloaders[set_name] = dataloader_class(dataset, 
                                                 batch_size=batch_size,
                                                 shuffle=shuffle,
                                                 **dataloader_kwargs)
    return dataloaders


"""
``get_inv_class_wts()``
-----------------------
Calculates inverse class weights from a set of training
labels. Can be used with nn.CrossEntropyLoss(weight=.),
for example, to help with training a class-imbalanced
dataset.
"""
from collections import Counter

def get_inv_class_wts(train_labels):
    cts = Counter(sorted(train_labels))
    inv_wts = [1 - (ct / len(train_labels)) for (label, ct) in cts.items()]
    return inv_wts


"""
``EpochCounter`` class
----------------------
Contains a class `Epoch` with state_dict implementations,
for use in saving and loading model states for continuing
training, etc. Useful with the accelerate library.
"""
class EpochCounter:
    def __init__(self, 
                 n: int = 0,
                 metric_name: str = 'loss_valid'):
        self.n = n
        self.best = {
            metric_name: {
                'epoch': 0,
                'score': 0.0
            },
            # track valid loss separately, even if it's the metric of interest
            '_valid_loss': {
                'epoch': 0,
                'score': 1.0e24
            }
        }

    def __iadd__(self, m):
        self.n = self.n + m
        return self

    def state_dict(self):
        return self.__dict__

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            setattr(self, k, v)

    def set_best(self, metric, epoch, score):
        if metric in self.best.keys():
            self.best[metric]['epoch'] = epoch
            self.best[metric]['score'] = score
        else:
            self.best[metric] = {
                'epoch': epoch,
                'score': score
            }

    def __str__(self):
        return str(self.n)

