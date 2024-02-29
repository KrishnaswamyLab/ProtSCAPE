from __future__ import print_function, division
import os
from networkx.drawing import nx_pylab
from networkx.relabel import convert_node_labels_to_integers
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch_geometric.data
import pickle
from models.LEGS_module import Scatter
from graphein.ml.conversion import GraphFormatConvertor
from graphein.protein.features.nodes.amino_acid import meiler_embedding
from graphein.protein.features.nodes.dssp import add_dssp_feature, add_dssp_df
from graphein.protein.config import ProteinGraphConfig

#from torch_geometric.utils.convert import from_networkx

import networkx as nx

class DEShaw(Dataset):
    """ZINCTranch dataset."""

    def __init__(self, file_name, transform=None, dssp=False):
        with open(file_name, 'rb') as file:
            self.graphs = pickle.load(file)
            self.dssp = dssp
        
        self.amino_acid_dict = {'MET' : 0,
                                'GLN' : 1,
                                'TYR' : 2,
                                'LYS' : 3,
                                'LEU' : 4,
                                'VAL' : 5,
                                'ILE' : 6,
                                'ASN' : 7,
                                'GLY' : 8,
                                'THR' : 9,
                                'GLU' : 10,
                                'ALA' : 11,
                                'ASP' : 12,
                                'PHE' : 13,
                                'TRP' : 14
        }
        self.num_node_features = len(self.amino_acid_dict.keys())
        self.transform = transform

    def __len__(self):
        
        return len(self.graphs)

    def __getitem__(self, idx):

        index_dict = {'0' : '0 to 2 us/',\
                      '1' : '2 to 4 us/',\
                      '2' : '4 to 6 us/',\
                      '3' : '6 to 8 us/',\
                      '4' : '8 to 10 us/'       
        }
             
        data = self.graphs[idx]
        # print(data.name)
        a, b = data.name.split('_')
        a, val = b.split('-')
        dir = index_dict[a]

        dir = "/data/lab/de_shaw/all_trajectory_slices/GB3/" + dir + data.name[0] + '.pdb'

        converter = GraphFormatConvertor(src_format="pyg", dst_format="nx", verbose="gnn", columns=None)
        nx_graph = converter.convert_pyg_to_nx(data)
        # config = ProteinGraphConfig(pdb_dir="/data/lab/de_shaw/all_trajectory_slices/GB3/0 to 2 us")
        # nx_graph.graph['config'] = config
        # nx_graph.graph['pdb_id'] = data.name[0]


        feats = []


        if self.dssp:
            nx_graph = add_dssp_df(nx_graph)
            nx_graph = add_dssp_feature(nx_graph, "phi")
            nx_graph = add_dssp_feature(nx_graph, "rsa")

        else :
            nodes = data.node_id

            for i in range(data.num_nodes):
                arr = np.zeros(15)
                acid = nodes[i]
                a, acid, index = acid.split(':')
                index = self.amino_acid_dict[acid]
                arr[index] = 1

                feats.append(arr)
        data.x = torch.tensor(feats).float()
        data.edge_attr = None
        a, b = data.name.split('_')
        a, val = b.split('-')
        #int(val) is 0-2 microseconds
        val = int(val) + int(a) * 10000
        #Time value
        data.time = val
        # print(data.x)
        if self.transform: 
            # print(data)
            return self.transform(data)
        else:
            return data

class Scattering(object):

    def __init__(self):
        model = Scatter(15, trainable_laziness=None)
        model.eval()
        self.model = model
    
    def __call__(self, sample):

        with torch.no_grad():
            to_return = self.model(sample)
        # print(to_return[0])
        return to_return[0], torch.tensor([float(sample.time)])

class Meiler_Embedding(object):

    def __call__(self, sample):

        print(sample.x)

        