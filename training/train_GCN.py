import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from argparse import ArgumentParser
import pickle
from models.GCN import GCN, train, test

if __name__ == "__main__":
    parser = ArgumentParser()

    # parser.add_argument('--input_dim', default=None, type=int)
    parser.add_argument('--hidden_dim', default=64, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--protein', default=None, type=str)
    args = parser.parse_args()


    #55 residues
    if args.protein == '1bx7':
        #Change to analyis of 1bgf_A_protein
        with open('1bx7_A_analysis/graphsrog.pkl', 'rb') as file:
            full_dataset =  pickle.load(file)
    #46 residues
    if args.protein == '1ab1':
        with open('1ab1_A_analysis(1)/graphsrog.pkl', 'rb') as file:
            full_dataset =  pickle.load(file)
    #60 residues
    if args.protein == '1bxy':
        with open('1bxy_A_analysis/graphsrog.pkl', 'rb') as file:
            full_dataset =  pickle.load(file)
    
    
    # import pdb; pdb.set_trace()
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    # import pdb; pdb.set_trace()
    # train loader
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                                        shuffle=True, num_workers=15)
    # valid loader 
    valid_loader = DataLoader(val_set, batch_size=args.batch_size,
                                        shuffle=False, num_workers=15)
    
    model = GCN(num_features=20, hidden_size=args.hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    train(model, train_loader, optimizer, criterion)

    # Test the model
    test(model, valid_loader)
