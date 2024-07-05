import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm
from torch_geometric.nn import GINConv
import mdtraj as md
from baselines.Baseline_1.metrics.metrics_fns import calc_dope_scores
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
RMSD functions
"""
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
    import mdtraj as md

    
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

class GIN(nn.Module):
    def __init__(self, num_features, hidden_size):
        super(GIN, self).__init__()
        nn1 = nn.Sequential(nn.Linear(num_features, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size))
        nn2 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size))
        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)
        self.fc = nn.Linear(hidden_size, 3)  # Output 3D coordinates

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc(x)
        return x


def train(model, loader, optimizer, criterion, residue_num, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for data in loader:
            optimizer.zero_grad()
            # import pdb; pdb.set_trace()
            output = model(data.x, data.edge_index)
            # import pdb; pdb.set_trace()
            output = output.view(data.time.shape[0], residue_num, 3)
            y_true = torch.tensor(data.coords, dtype=torch.float32)
            y_true = y_true.reshape(data.time.shape[0], residue_num, 3)
            # import pdb; pdb.set_trace()
            loss = criterion(output, y_true)  # Use data.coords for 3D coordinates
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
        avg_loss = total_loss / len(loader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

def test(model, loader, val_set, residue_num):
    model.eval()
    coords_recon_lst = []
    for data in tqdm(loader):
        output = model(data.x, data.edge_index)
        output = output.view(data.time.shape[0], residue_num, 3)
        # loss = F.l1_loss(output, data.y)  # Using L1 loss for absolute error
        coords_recon_lst.append(output.detach())
    coords_recon = torch.cat(coords_recon_lst, dim=0)
    coords_gt = np.array([data.coords for data in val_set])
    pcc_lst = []
    scc_lst = []
    for i in range(coords_recon.shape[0]):
        pcc, scc = eucl_dist_corrs(coords_recon[i], coords_gt[i], squared=False)

        pcc_lst.append(pcc)
        scc_lst.append(scc.correlation)
    pcc_mean = np.mean(pcc_lst)
    # pcc_std = np.std(pcc_lst)
    scc_mean = np.mean(scc_lst)
    # scc_std = np.std(scc_lst)
    return pcc_mean, scc_mean
    # print(f'PCC: {np.mean(pcc_lst):.4f}, SCC: {np.mean(scc_lst):.4f}')

def test_rmsd(model, loader, val_set, residue_num, traj):
    model.eval()
    coords_recon_lst = []
    for data in tqdm(loader):
        output = model(data.x, data.edge_index)
        output = output.view(data.time.shape[0], residue_num, 3)
        coords_recon_lst.append(output.detach())
    coords_recon = torch.cat(coords_recon_lst, dim=0)
    coords_gt = np.array([data.coords for data in val_set])
    # print('coords_recon:', coords_recon.shape)
    # print('coords_gt:', coords_gt.shape)
    rmsd_lst = []
    for i in range(coords_recon.shape[0]):
        ref_frame =  traj[i]
        ref_frame_residue_coords = get_residue_coords(ref_frame)    
        atomic_frame = est_atomic_pdb_from_residue_coords(
            orig_frame=ref_frame, 
            new_residue_coords=coords_recon[i].numpy(),
            orig_residue_coords=ref_frame_residue_coords
            )
        rmsd = md.rmsd(target=atomic_frame, reference=ref_frame)
        rmsd_lst.append(rmsd)
    rmsd_mean = np.mean(rmsd_lst)
    print(f'RMSD: {rmsd_mean:.4f}')
    return rmsd_mean

def test_dope(model, loader, val_set, residue_num, traj, tmp_pdb_savepath):
    model.eval()
    coords_recon_lst = []
    for data in tqdm(loader):
        output = model(data.x, data.edge_index)
        output = output.view(data.time.shape[0], residue_num, 3)
        coords_recon_lst.append(output.detach())
    coords_recon = torch.cat(coords_recon_lst, dim=0)
    coords_gt = np.array([data.coords for data in val_set])
    # print('coords_recon:', coords_recon.shape)
    # print('coords_gt:', coords_gt.shape)
    dope_list = []
    for i in range(coords_recon.shape[0]):
        ref_frame =  traj[i]
        ref_frame_residue_coords = get_residue_coords(ref_frame)    
        atomic_frame = est_atomic_pdb_from_residue_coords(
            orig_frame=ref_frame, 
            new_residue_coords=coords_recon[i].numpy(),
            orig_residue_coords=ref_frame_residue_coords
            )
        dope = calc_dope_scores(atomic_frame.xyz[0], ref_frame, tmp_pdb_savepath=tmp_pdb_savepath, normalize=True, verbosity=0)
        dope_list.append(dope)
    dope_mean = np.mean(dope_list)
    print(f'DOPE: {dope_mean:.4f}')
    return dope_mean