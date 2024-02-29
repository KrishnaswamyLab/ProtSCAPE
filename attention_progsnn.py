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

from models.gsae_model import GSAE
from models.progsnn import ProGSNN
from torch_geometric.loader import DataLoader
from torchvision import transforms

from deshaw_processing.de_shaw_Dataset import DEShaw, Scattering


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--dataset', default='deshaw', type=str)

    parser.add_argument('--input_dim', default=None, type=int)
    parser.add_argument('--latent_dim', default=128, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--embedding_dim', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)

    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--beta', default=0.0005, type=float)
    parser.add_argument('--n_epochs', default=40, type=int)
    parser.add_argument('--len_epoch', default=None)
    parser.add_argument('--probs', default=0.2)
    parser.add_argument('--nhead', default=1)
    parser.add_argument('--layers', default=1)
    parser.add_argument('--task', default='reg')
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--n_gpus', default=1, type=int)
    parser.add_argument('--save_dir', default='train_logs/', type=str)

    # add args from trainer
    # parser = pl.Trainer.add_argparse_args(parser)
    # parse params 
    args = parser.parse_args()


    full_dataset = DEShaw('graphs/total_graphs.pkl')
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    # print(len(full_dataset))
    # train loader
    # import pdb; pdb.set_trace()
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                                        shuffle=True, num_workers=15)
    # valid loader 
    valid_loader = DataLoader(val_set, batch_size=args.batch_size,
                                        shuffle=False, num_workers=15)
    
    full_loader = DataLoader(full_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=15)

   
    early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience=5,
            verbose=True,
            mode='min'
            )
   
    args.input_dim = full_dataset[0].x.shape[-1]
    # print(args.input_dim)
    
    # print(train_set[0].x.shape[-1])
    # print(full_dataset[0][0].shape)
    args.prot_graph_size = 660
    args.len_epoch = len(full_loader)
    # init module
    model = ProGSNN(args)
    trained_weights = torch.load('/home/sv496/project/ProGSNN-1/train_logs/progsnn_logs_run_deshaw_2024-01-22-18/model.npy')
    model.load_state_dict(trained_weights)
    # model.eval()
    # import pdb; pdb.set_trace()
    residual_attention = []
    embeddings = []
    print(len(full_dataset))
    with torch.no_grad():
        loss = model.get_loss_list()
        for batch in full_loader:
            y_pred, z_rep, coeffs, coeffs_recon, attention_maps, att_maps_res = model(batch)
            # print(attention_maps[0].shape)
            # print(att_maps_res[0].shape)
            
            residual_attention.append(att_maps_res[0])
            
            embeddings.append(z_rep)

    # print('saving reconstruction loss')
    # loss = np.array(loss)
    # np.save(save_dir + "reg_loss_list.npy", loss)

    print('saving attention map')
    # residual_attention = np.stack(residual_attention)
    # np.save(save_dir + "attention_maps.npy", residual_attention)
    with open('attention_residue.pkl', 'wb') as file:
        pickle.dump(residual_attention, file)

    # print('saving embeddings')
    # embeddings = np.array(embeddings)
    # np.save(save_dir + "embeddings.npy", embeddings)  

    



    # EVALUATION ON TEST SET 
    # energy pred mse

    #print("getting test set predictions")
    #with torch.no_grad():
    #    x_recon_test = model(test_tup[0])[0]
    #    y_pred_test = model.predict_from_data(test_tup[0])


    # print("adj type: {}".format(test_tup[1].flatten().numpy()))
    # print("adj_hat type: {}".format(adj_hat_test.flatten().detach().numpy()))

    #recon_test_val = nn.MSELoss()(x_recon_test.flatten(), test_tup[0].flatten())
    #pred_test_val = nn.MSELoss()(y_pred_test.flatten(), test_tup[-1].flatten())

    #print("logging test set metrics")
    # wandb_logger.log_metrics({'test_recon_MSE':recon_test_val,
    #                             'test_pred_MSE': pred_test_val})


    #print("gathering eval subsets")
    #eval_tup_list = [eval_metrics.compute_subsample([test_embed, test_tup[-1]], 10000)[0] for i in range(8)]
    # trainer.test()
    #print("getting smoothness vals")
    #embed_eval_array= np.expand_dims(np.array([x[0].numpy() for x in eval_tup_list]),0)
    #energy_eval_array= np.array([x[1].numpy() for x in eval_tup_list])

    #print('embed_eval_array shape: {}'.format(embed_eval_array.shape))
    #print('energy_eval_array shape: {}'.format(energy_eval_array.shape))
    
    # energy_smoothness = eval_metrics.eval_over_replicates(embed_eval_array,
    #                                                         energy_eval_array,
    #                                             eval_metrics.get_smoothnes_kNN,
    #                                             [5, 10])[0]

    # energy_smoothness = eval_metrics.format_metric(energy_smoothness)


    # wandb_logger.log_metrics({'e_smooth_k5_mean':energy_smoothness[0][0],
    #                             'e_smooth_k10_mean': energy_smoothness[0][1],
    #                             'e_smooth_k5_std': energy_smoothness[1][0],
    #                             'e_smooth_k10_std': energy_smoothness[1][1]})