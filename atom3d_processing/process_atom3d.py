import atom3d.datasets.datasets as da
from load_atom3d import ProtGraphTransform, dev_prot_df_to_graph, Rg
from tqdm import tqdm
from atom3d.datasets import LMDBDataset
from torch_geometric.data import Data
import pickle
class Atom3dLoader:
    def __init__(self, dataset_path):
        self.dataset = dataset_path
    def load_data(self):
        full_dataset = LMDBDataset(self.dataset)
        return full_dataset
    def progsnn_loader(self, full_dataset, data, property= None):
        dataset = []
        if data == "msp":
            # import pdb; pdb.set_trace()
            for x in tqdm(full_dataset):
                item = x['mutated_atoms']
                
                if property == 'Rg':
                    rg = Rg(item)
                    node_feats, edge_index, edge_feats, pos = dev_prot_df_to_graph(item,feat_col='resname')
                    graph = Data(node_feats, edge_index, edge_feats, y=rg, pos=pos)
                    dataset.append(graph)
                else:
                    node_feats, edge_index, edge_feats, pos = dev_prot_df_to_graph(item,feat_col='resname')
                    graph = Data(node_feats, edge_index, edge_feats, y=x['label'], pos=pos)
                    dataset.append(graph)

        return dataset
    def VAE_loader(self, full_dataset, data, property):
        pass

if __name__ == '__main__':
    data = Atom3dLoader('../data/raw/MSP/data/')
    full_data = data.load_data()
    data = data.progsnn_loader(full_data, data='msp')
    with open('data_msp.pk', 'wb') as f:
        pickle.dump(data, f)