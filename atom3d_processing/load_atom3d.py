import math
import numpy as np
import scipy.spatial as ss
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
import atom3d.datasets.datasets as da
import pandas as pd


import pathlib
# from ProGSNN.utils.data_utils import load_pickle_file

# PDB atom names -- these include co-crystallized metals
prot_atoms = ['C', 'H', 'O', 'N', 'S', 'P', 'ZN', 'NA', 'FE', 'CA', 'MN', 'NI', 'CO', 'MG', 'CU', 'CL', 'SE', 'F']
# RDKit molecule atom names
mol_atoms = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
             'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
             'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',  # H?
             'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
             'Cr', 'Pt', 'Hg', 'Pb']
# Residue names
residues = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG',
                'SER', 'THR', 'VAL', 'TRP', 'TYR']



one_hot_dict = {'element': prot_atoms, 'resname': residues}


class ProtGraphTransform(object):
    def __init__(self, atom_keys, label_key):
        self.atom_keys = atom_keys
        self.label_key = label_key
    

        print(self.atom_keys)

    def __call__(self, item):
        # transform protein and/or pocket to PTG graphs
        item = dev_prot_graph_transform(item, atom_keys=[self.atom_keys], label_key=self.label_key)
        
        return item
        
        
def dev_prot_graph_transform(item, atom_keys, label_key, feat_col):
    """Transform for converting dataframes to Pytorch Geometric graphs, to be applied when defining a :mod:`Dataset <atom3d.datasets.datasets>`.
    Operates on Dataset items, assumes that the item contains all keys specified in ``keys`` and ``labels`` arguments.
    :param item: Dataset item to transform
    :type item: dict
    :param atom_keys: list of keys to transform, were each key contains a dataframe of atoms, defaults to ['atoms']
    :type atom_keys: list, optional
    :param label_key: name of key containing labels, defaults to ['scores']
    :type label_key: str, optional
    :return: Transformed Dataset item
    :rtype: dict
    """    
    from torch_geometric.data import Data
    import atom3d.util.graph as gr

    for key in atom_keys:
        node_feats, edge_index, edge_feats, pos = dev_prot_df_to_graph(item[key], feat_col=feat_col)
        item[key] = Data(node_feats, edge_index, edge_feats, y=item[label_key], pos=pos)

    return item


def dev_prot_df_to_graph(df, feat_col, edge_dist_cutoff=6.0):
    r"""
    Converts protein in dataframe representation to a graph compatible with Pytorch-Geometric, where each node is an atom.
    :param df: Protein structure in dataframe format.
    :type df: pandas.DataFrame
    :param node_col: Column of dataframe to find node feature values. For example, for atoms use ``feat_col="element"`` and for residues use ``feat_col="resname"``
    :type node_col: str, optional
    :param allowable_feats: List containing all possible values of node type, to be converted into 1-hot node features. 
        Any elements in ``feat_col`` that are not found in ``allowable_feats`` will be added to an appended "unknown" bin (see :func:`atom3d.util.graph.one_of_k_encoding_unk`).
    :type allowable_feats: list, optional
    :param edge_dist_cutoff: Maximum distance cutoff (in Angstroms) to define an edge between two atoms, defaults to 4.5.
    :type edge_dist_cutoff: float, optional
    :return: tuple containing
        - node_feats (torch.FloatTensor): Features for each node, one-hot encoded by values in ``allowable_feats``.
        - edges (torch.LongTensor): Edges in COO format
        - edge_weights (torch.LongTensor): Edge weights, defined as a function of distance between atoms given by :math:`w_{i,j} = \frac{1}{d(i,j)}`, where :math:`d(i, j)` is the Euclidean distance between node :math:`i` and node :math:`j`.
        - node_pos (torch.FloatTensor): x-y-z coordinates of each node
    :rtype: Tuple
    """ 

    allowable_feat = one_hot_dict[feat_col]
    def first_resname(series):
        return series.iloc[0]
    #Average the x, y, z coordinates over the atoms of the same residue
    # import pdb; pdb.set_trace()
    # Separate data for chains A and B
    chain_a = df[df['chain'] == 'A']
    chain_b = df[df['chain'] == 'B']

    avg_coords_a = chain_a.groupby('residue').mean().reset_index()
    avg_coords_a = pd.merge(avg_coords_a, chain_a[['residue', 'resname']], on='residue').drop_duplicates()

    # Calculate average coordinates for each residue in chain B and merge with 'resname'
    avg_coords_b = chain_b.groupby('residue').mean().reset_index()
    avg_coords_b = pd.merge(avg_coords_b, chain_b[['residue', 'resname']], on='residue').drop_duplicates()

    # Concatenate the results into a single dataframe
    df = pd.concat([avg_coords_a, avg_coords_b], ignore_index=True)
    # import pdb; pdb.set_trace()
    # df = df.groupby('residue').agg({
    #     'resname': first_resname,
    #     'subunit': 'mean',
    #     'model': 'mean',
    #     'occupancy': 'mean',
    #     'bfactor': 'mean',
    #     'x': 'mean',
    #     'y': 'mean',
    #     'z': 'mean',
    #     'serial_number': 'mean',
    #     'element': lambda x: list(x)
    # }).reset_index()
    # import pdb; pdb.set_trace()
    
    node_pos = torch.FloatTensor(df[['x', 'y', 'z']].to_numpy())

    kd_tree = ss.KDTree(node_pos)
    edge_tuples = list(kd_tree.query_pairs(edge_dist_cutoff))
    edges = torch.LongTensor(edge_tuples).t().contiguous()
    edges = to_undirected(edges)

    node_feats = torch.FloatTensor([one_of_k_encoding_unk(e, allowable_feat) for e in df[feat_col]])
    edge_weights = torch.FloatTensor(
        [1.0 / (np.linalg.norm(node_pos[i] - node_pos[j]) + 1e-5) for i, j in edges.t()]).view(-1)
    # feats = F.one_hot(elems, num_classes=len(atom_int_dict))
    
    return node_feats, edges, edge_weights, node_pos


def one_of_k_encoding_unk(x, allowable_set):
    """Converts input to 1-hot encoding given a set of allowable values. Additionally maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

#### nn loading utils 

"""
Function to compute Radius of Gyration: https://github.com/sarisabban/Rg/blob/master/Rg.py
	Calculates the Radius of Gyration (Rg) of a protein given its .pdb 
	structure file. Returns the Rg integer value in Angstrom.

"""

def Rg(df):
	
    coord = list()
    mass = list()
    for index, atom in df.iterrows():
        # import pdb; pdb.set_trace()
        x = atom['x']
        y = atom['y']
        z = atom['z']
        coord.append([x, y, z])
        if atom['element'] == 'C':
            mass.append(12.0107)
        elif atom['element'] == 'O':
            mass.append(15.9994)
        elif atom['element'] == 'N':
            mass.append(14.0067)
        elif atom['element'] == 'S':
            mass.append(32.065)
    xm = [(m*i, m*j, m*k) for (i, j, k), m in zip(coord, mass)]
    tmass = sum(mass)
    rr = sum(mi*i + mj*j + mk*k for (i, j, k), (mi, mj, mk) in zip(coord, xm))
    mm = sum((sum(i) / tmass)**2 for i in zip(*xm))
    rg = math.sqrt(rr / tmass-mm)
    return(round(rg, 3))









    


