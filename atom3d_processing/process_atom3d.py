import atom3d.datasets.datasets as da
from load_atom3d import ProtGraphTransform, dev_prot_df_to_graph, Rg
from tqdm import tqdm
from atom3d.datasets import LMDBDataset
from torch_geometric.data import Data
import pickle
import warnings
# Ignore all warnings temporarily
warnings.filterwarnings("ignore")
class Atom3dLoader:
    def __init__(self, dataset_path):
        self.dataset = dataset_path
    def load_data(self):
        full_dataset = LMDBDataset(self.dataset)
        return full_dataset
    def progsnn_loader(self, full_dataset, data, property= None):
        dataset = []
        if data == "msp":
            # diff = full_dataset[0]['mutated_atoms'][full_dataset[0]['mutated_atoms'].ne(full_dataset[1]['mutated_atoms']).any(axis=0)]
            # import pdb; pdb.set_trace()
            for x in tqdm(full_dataset):
                item = x['mutated_atoms']
                # print(item['element'][2497])
            
                # has_na = item.isna().any().any()
                # # Check for 0 values
                # has_zero = (item == 0).any().any()
                # if has_na:
                #     print("AYO WHAT!")
                # import pdb; pdb.set_trace()
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