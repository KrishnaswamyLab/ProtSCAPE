import atom3d.datasets.datasets as da
from load_atom3d import ProtGraphTransform, dev_prot_df_to_graph
from tqdm import tqdm
from atom3d.datasets import LMDBDataset
from torch_geometric.data import Data
import pickle
def load_data():
    # full_dataset = da.load_dataset('data/msp/raw/MSP/data/', transform=ProtGraphTransform(atom_keys='original_atoms', label_key='label'))
    full_dataset = LMDBDataset('data/psr/raw/casp5_to_13/data/')
    dataset = []
    
    for x in tqdm(full_dataset):
        item = x['atoms']
        node_feats, edge_index, edge_feats, pos = dev_prot_df_to_graph(item,feat_col='element')
        graph = Data(node_feats, edge_index, edge_feats, y=x['scores'], pos=pos)
        dataset.append(graph)
    
    return dataset


if __name__ == '__main__':
    data = load_data()
    with open('data_psr.pk', 'wb') as f:
        pickle.dump(data, f)
    # data_to_graph(data)