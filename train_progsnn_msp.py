from argparse import ArgumentParser
import datetime
import os
import numpy as np

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
from atom3d.util.metrics import auroc
from models.gsae_model import GSAE
from models.progsnn import ProGSNN
from torch_geometric.loader import DataLoader
from torchvision import transforms
from atom3d.datasets import LMDBDataset
from deshaw_processing.de_shaw_Dataset import DEShaw, Scattering


if __name__ == '__main__':
    # torch.cuda.is_available()
    # import pdb; pdb.set_trace()
    parser = ArgumentParser()

    parser.add_argument('--dataset', default='deshaw', type=str)

    parser.add_argument('--input_dim', default=None, type=int)
    parser.add_argument('--latent_dim', default=100, type=int)
    parser.add_argument('--hidden_dim', default=200, type=int)
    parser.add_argument('--embedding_dim', default=80, type=int)
    parser.add_argument('--lr', default=0.001, type=float)

    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--beta_loss', default=0.5, type=float)
    parser.add_argument('--beta', default=0.0005, type=float)
    parser.add_argument('--n_epochs', default=10, type=int)
    parser.add_argument('--len_epoch', default=None)
    parser.add_argument('--probs', default=0.2)
    parser.add_argument('--nhead', default=1)
    parser.add_argument('--layers', default=1)
    parser.add_argument('--task', default='bin_class')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--n_gpus', default=1, type=int)
    parser.add_argument('--save_dir', default='train_logs/', type=str)

    # add args from trainer
    # parser = pl.Trainer.add_argparse_args(parser)
    # parse params 
    args = parser.parse_args()
    torch.cuda.empty_cache()

    # full_dataset = DEShaw('graphs/total_graphs.pkl')
    with open('data_msp.pk', 'rb') as file:
        full_dataset =  pickle.load(file)
    
    # full_dataset = LMDBDataset('data/msp/raw/MSP/data/')
    # import pdb; pdb.set_trace()
    # full_dataset = [x for x in full_dataset if x.num_nodes < 1000]
    # print(len(full_dataset))
    #Convert the list of 0s and 1s target strings to integers and a torch tensor FOR MSP PREDICTIONS
    for data in full_dataset:
        y = torch.tensor([int(label) for label in data.y]).float()
        data.y = y
    
    # import pdb; pdb.set_trace()
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    # print(len(full_dataset))
    # import pdb; pdb.set_trace()
    # train loader
    train_loader = DataLoader(train_set, batch_size=1)
    # valid loader 
    valid_loader = DataLoader(val_set, batch_size=1)
    # full_loader = DataLoader(full_dataset,
    #                          batch_size=args.batch_size,
    #                          shuffle=False,
    #                          num_workers=15)

    # logger
    # import pdb; pdb.set_trace()
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
            monitor='train_loss',
            min_delta=0.00,
            patience=5,
            verbose=True,
            mode='min'
            )
    # print(len(val_set))
    # args.input_dim = len(train_set)
    # print()
    args.input_dim = train_set[0].x.shape[-1]
    # args.input_dim = 20
    # print(train_set[0].x.shape[-1])

    # print(full_dataset[0][0].shape)
    # args.prot_graph_size = 1928
    args.prot_graph_size = max(
            [item.edge_index.shape[1] for item in full_dataset])
    args.len_epoch = len(train_loader)
    # init module
    model = ProGSNN(args)
    # model.half()
    print("Training model..")
    # most basic trainer, uses good defaults
    trainer = pl.Trainer(
                        max_epochs=args.n_epochs,
                        devices = "auto",
                        accelerator="gpu",
                        callbacks=[early_stop_callback],
                        logger = wandb_logger
                        )
    trainer.fit(model=model,
                train_dataloaders=train_loader,
                #val_dataloaders=valid_loader,
                )

    model = model.cpu()
    model.dev_type = 'cpu'
    print('saving model')
    torch.save(model.state_dict(), "model_MSP_RG.npy")
    model = model.eval()