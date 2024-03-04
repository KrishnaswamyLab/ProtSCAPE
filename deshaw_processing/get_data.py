import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import pytorch_lightning as pl
from tqdm.notebook import tqdm
import networkx as nx
#from torch_geometric.data import Data
#from torch_geometric.utils import from_networkx
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import f1_score
import pickle
import warnings
from tqdm import tqdm
import os


range_to_filename = {'0 to 2 us': 'pdgs0to2',
                    '2 to 4 us': 'pdgs2to4',
                    '4 to 6 us': 'pdgs4to6',
                    '6 to 8 us': 'pdgs6to8',
                    '8 to 10 us': 'pdgs8to10'}

range_to_graphname = {'0 to 2 us': '0to2graphs',
                    '2 to 4 us': '2to4graphs',
                    '4 to 6 us': '4to6graphs',
                    '6 to 8 us': '6to8graphs',
                    '8 to 10 us': '8to10graphs'}


print("getting graphs")
for entry in tqdm(range_to_filename.keys()):
    #warnings.filterwarnings("ignore")
    pdbs = np.load(f"../file_names_/{range_to_filename[entry]}.npy")
    # df = pd.read_csv("../datasets/pscdb/structural_rearrangement_data.csv")

    # pdbs = df["Free PDB"]
    #y = [torch.argmax(torch.Tensor(lab)).type(torch.LongTensor) for lab in LabelBinarizer().fit_transform(df.motion_type)]

    from graphein.protein.config import ProteinGraphConfig
    from graphein.protein.edges.distance import add_hydrogen_bond_interactions, add_peptide_bonds, add_k_nn_edges
    from graphein.protein.graphs import construct_graph

    from functools import partial

    # Override config with constructors
    constructors = {
        "edge_construction_functions": [partial(add_k_nn_edges, k=3, long_interaction_threshold=0)],
        "pdb_dir": f"/gpfs/gibbs/pi/krishnaswamy_smita/de_shaw/BPTI/{entry}",
        #"edge_construction_functions": [add_hydrogen_bond_interactions, add_peptide_bonds],
        #"node_metadata_functions": [add_dssp_feature]
    }

    config = ProteinGraphConfig(**constructors)

    # Make graphs
    graph_list = []
    y_list = []
    for idx, pdb in enumerate(tqdm(pdbs)):
        path = f"/gpfs/gibbs/pi/krishnaswamy_smita/de_shaw/BPTI/{entry}" + '/' + pdb
        # construct_graph(pdb_path=path,
        #                     config=config
        #                 )
        try:
            print(path)
            graph_list.append(
                construct_graph(path=path,
                            config=config
                        )
                )
            print("Appended!")
            #y_list.append(y[idx])
        except:
            print(str(idx) + ' processing error...')
            break
            pass

    from graphein.ml.conversion import GraphFormatConvertor

    format_convertor = GraphFormatConvertor('nx', 'pyg',
                                            verbose = 'gnn',
                                            columns = None)

    #pyg_list = [format_convertor(graph) for graph in tqdm(graph_list)]


    pyg_list = [format_convertor(graph) for graph in tqdm(graph_list)]
    # for i in pyg_list:
    #     if i.coords.shape[0] == len(i.node_id):
    #         pass
    #     else:
    #         print(i)
    #         pyg_list.remove(i)
    with open(f"graphs_bpti/{range_to_graphname[entry]}.pkl", "wb") as file:
        pickle.dump(pyg_list, file)

print("combining data")
total_graphs = []
arr = os.listdir("./graphs_bpti")
for i, entry in enumerate(tqdm(arr)):

    split1, split2 = entry.split('.')
    if split2 == 'pkl' and split1 != 'total_graphs':

        with open("./graphs_bpti/" + entry, "rb") as file:
            graphs = pickle.load(file)
            for graph in graphs:
                total_graphs.append(graph)

with open("./graphs_bpti/total_graphs.pkl", 'wb') as out:
    pickle.dump(total_graphs, out)