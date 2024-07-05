import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from argparse import ArgumentParser
import pickle
from models.GIN import GIN, train, test, test_rmsd, test_dope
import numpy as np
from numpy.random import RandomState
import mdtraj as md
import os
from deshaw_processing.de_shaw_Dataset import DEShaw, Scattering

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

idx_l = get_cv_idx_l(seed=CV_SEED_GB3,
                        dataset_size=1001, 
                        k=5)
pearson_all = []
spearman_all = []
rmsd_all = []
dope_all = []
for fold_i in range(5):
    parser = ArgumentParser()

    # parser.add_argument('--input_dim', default=None, type=int)
    parser.add_argument('--hidden_dim', default=64, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--protein', default=None, type=str)
    args = parser.parse_args()


    #55 residues
    if args.protein == '1bx7':
        residue_num = 55
        #Change to analyis of 1bgf_A_protein
        traj = md.load("1bx7_A_analysis/1bx7_A_R1.xtc", top= "1bx7_A_analysis/1bx7_A.pdb")
        tmp_pdb_savepath=f"1bx7_tmp_pdb/"
        with open('1bx7_A_analysis/graphsrog.pkl', 'rb') as file:
            full_dataset =  pickle.load(file)
    #50 residues
    if args.protein == '1ptq':
        residue_num = 50
        traj = md.load('1ptq_A_analysis/1ptq_A_R1.xtc', top='1ptq_A_analysis/1ptq_A.pdb')
        tmp_pdb_savepath=f"1ptq_tmp_pdbs/"
        with open('1ptq_A_analysis/graphsrog.pkl', 'rb') as file:
            full_dataset =  pickle.load(file)
        
    #60 residues
    if args.protein == '1bxy':
        residue_num = 60
        traj = md.load("1bxy_A_analysis/1bxy_A_R1.xtc", top= "1bxy_A_analysis/1bxy_A.pdb")
        tmp_pdb_savepath=f"1bxy_tmp_pdb/"
        with open('1bxy_A_analysis/graphsrog.pkl', 'rb') as file:
            full_dataset =  pickle.load(file)
    
    if args.protein == 'gb3':
        deshaw_records = get_deshaw_data_info("/gpfs/gibbs/pi/krishnaswamy_smita/de_shaw/GB3")
        traj = md.load(deshaw_records['pdb_filepaths'])
        tmp_pdb_savepath = f"gb3_tmp_pdbs/"
        full_dataset = DEShaw('deshaw_processing/graphs_gb3/total_graphs.pkl')
        residue_num = full_dataset[0].x.shape[0]
    # full_dataset = DEShaw('deshaw_processing/graphs_gb3/total_graphs.pkl')
    if args.protein == 'bpti':
        # print("GOING INTO BPTI")
        deshaw_records = get_deshaw_data_info("/gpfs/gibbs/pi/krishnaswamy_smita/de_shaw/BPTI")
        traj = md.load(deshaw_records['pdb_filepaths'])
        tmp_pdb_savepath = f"bpti_tmp_pdbs/"
        full_dataset = DEShaw('deshaw_processing/graphs_bpti/total_graphs.pkl')
        residue_num = full_dataset[0].x.shape[0]
    
    
    # import pdb; pdb.set_trace()
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
    
    model = GIN(num_features=20, hidden_size=args.hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    print("Training..")
    train(model, train_loader, optimizer, criterion, residue_num)

    # Test the model
    print("Testing..")
    dope_mean = test_dope(model, valid_loader, val_set, residue_num, traj, tmp_pdb_savepath)
    dope_all.append(dope_mean)
# dope_all = np.array(dope_all)
mean_dope = np.mean(dope_all, axis = 0)
std_dev_dope = np.std(dope_all, axis=0)
print(f'Mean DOPE: {mean_dope:.4f} ± {std_dev_dope:.4f}')




#     rmsd_mean = test_rmsd(model, valid_loader, val_set, residue_num, traj)
#     rmsd_all.append(rmsd_mean)

# rmsd_all = np.array(rmsd_all)
# mean_rmsd = np.mean(rmsd_all, axis = 0)
# std_dev_rmsd = np.std(rmsd_all, axis=0)
# print(f'Mean RMSD: {mean_rmsd:.4f} ± {std_dev_rmsd:.4f}')




#     pcc_mean, scc_mean = test(model, valid_loader, val_set, residue_num)
#     pearson_all.append(pcc_mean)
#     spearman_all.append(scc_mean)
# pearson_all = np.array(pearson_all)
# mean_pcc = np.mean(pearson_all, axis = 0)
# std_dev_pcc = np.std(pearson_all, axis=0)


# spearman_all = np.array(spearman_all)
# mean_scc = np.mean(spearman_all, axis = 0)
# std_dev_scc = np.std(spearman_all, axis=0)

# print(f'Mean PCC: {mean_pcc:.4f} ± {std_dev_pcc:.4f}')
# print(f'Mean SCC: {mean_scc:.4f} ± {std_dev_scc:.4f}')


