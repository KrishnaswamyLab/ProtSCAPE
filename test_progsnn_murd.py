from argparse import ArgumentParser
import datetime
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import pickle
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path

from models.gsae_model import GSAE
from models.progsnn import ProGSNN_ATLAS
from torch_geometric.loader import DataLoader
from torchvision import transforms

from deshaw_processing.de_shaw_Dataset import DEShaw, Scattering


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--dataset', default='deshaw', type=str)

    parser.add_argument('--input_dim', default=None, type=int)
    parser.add_argument('--latent_dim', default=64, type=int)
    parser.add_argument('--hidden_dim', default=64, type=int)
    parser.add_argument('--embedding_dim', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)

    parser.add_argument('--alpha', default=2, type=float)
    parser.add_argument('--beta', default=0.0005, type=float)
    parser.add_argument('--beta_loss', default=0.2, type=float)
    parser.add_argument('--gamma', default=0.0005, type=float)
    # parser.add_argument('--delta', default=0.0005, type=float)
    parser.add_argument('--n_epochs', default=40, type=int)
    parser.add_argument('--len_epoch', default=None)
    parser.add_argument('--probs', default=0.2)
    parser.add_argument('--nhead', default=1)
    parser.add_argument('--layers', default=1)
    parser.add_argument('--task', default='reg')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--n_gpus', default=1, type=int)
    parser.add_argument('--save_dir', default='train_logs/', type=str)
    parser.add_argument('--residue_num', default=None, type=int)
    parser.add_argument('--protein', default=None, type=str)
    # add args from trainer
    # parser = pl.Trainer.add_argparse_args(parser)
    # parse params 
    args = parser.parse_args()


    #55 residues
    if args.protein == 'murd':
        with open('MurD/graphs_MurD.pkl', 'rb') as file:
            full_dataset =  pickle.load(file)
        
    import pdb; pdb.set_trace()
    
    # full_dataset = full_dataset[:1000]

    # for data in full_dataset:
    #     y = float(data.y)
    #     data.y = y
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                                        shuffle=False, num_workers=15)
    # valid loader 
    valid_loader = DataLoader(val_set, batch_size=args.batch_size,
                                        shuffle=False, num_workers=15)
    full_loader = DataLoader(full_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=15)

    # logger
    now = datetime.datetime.now()
    date_suffix = now.strftime("%Y-%m-%d-%M")
    save_dir =  args.save_dir + 'progsnn_logs_run_{}_{}/'.format(args.dataset,date_suffix)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    wandb_logger = WandbLogger(name='run_progsnn',
                                project='progsnn', 
                                log_model=True,
                                save_dir=save_dir)
    
    wandb_logger.log_hyperparams(args)
    wandb_logger.experiment.log({"logging timestamp":date_suffix})

    # print(train_loader)
    # print([item for item in full_dataset])
    # early stopping 
    early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience=5,
            verbose=True,
            mode='min'
            )
    # import pdb; pdb.set_trace()
    args.input_dim = val_set[0].x.shape[-1]
    args.input_dim = 3
    print(val_set[0].x.shape[-1])
    # print(full_dataset[0][0].shape)
    args.prot_graph_size = max(
            [item.edge_index.shape[1] for item in full_dataset])
    print(args.prot_graph_size)
    # import pdb; pdb.set_trace()
    args.len_epoch = len(train_loader)

    args.residue_num = full_dataset[0].x.shape[0]
    # init module
    model = ProGSNN_ATLAS(args)

    # # most basic trainer, uses good defaults
    # trainer = pl.Trainer(
    #                     max_epochs=args.n_epochs,
    #                     #gpus=args.n_gpus,
    #                     callbacks=[early_stop_callback],
    #                     logger = wandb_logger
    #                     )
    # trainer.fit(model=model,
    #             train_dataloaders=train_loader,
    #             val_dataloaders=valid_loader,
    #             )


    #test model
    trained_weights = torch.load('train_logs/progsnn_logs_run_murd_2024-06-19-54/model_murd_murd.npy')
    model.load_state_dict(trained_weights)
    model = model.eval()
    attention_maps_col = []
    attention_maps_row = []
    # import pdb; pdb.set_trace()
    # get test set prediction
    times = np.array([data.time for data in full_dataset])
    test_latent = []
    latent_embeddings = []
    coords_recon_lst = []
    with torch.no_grad():
        for x in tqdm(full_loader):
            print("Looping through test set..")
            y_hat, z_rep, _, _, _, att_map_row,coords_recon, _, _ = model(x)
            # import pdb; pdb.set_trace()
           
            # attention_maps_col.append(att_map_col)
            # attention_maps_row.append(att_map_row)
            coords_recon_lst.append(coords_recon)
            test_latent.append(y_hat)
            latent_embeddings.append(z_rep)
    
    print(test_latent)
    test_predictions = torch.cat(test_latent, dim=0)
    
    # print("Saving attention maps..")
    # with open(f'attention_maps_{args.protein}.pkl', 'wb') as file:
    #     pickle.dump(attention_maps_row, file)
    
    print("Saving latent embeddings..")
    with open(f'latent_embeddings{args.protein}_5a5e.pkl', 'wb') as file:
        pickle.dump(latent_embeddings, file)
    
    print("Saving times..")
    with open(f'times_{args.protein}_5a5e.pkl', 'wb') as file:
        pickle.dump(times, file)
    
    # print("Saving coordinates..")
    # with open(f'coords_{args.protein}.pkl', 'wb') as file:
    #     pickle.dump(coords_recon_lst, file)
    