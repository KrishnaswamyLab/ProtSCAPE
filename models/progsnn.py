import argparse

import torch
import torch.nn as nn
from models.auxnetwork import str2auxnetwork
from models.bottleneck_progsnn import BaseBottleneck
from models.scatter import Scatter
from models.transformer import PositionalEncoding, TransformerEncoder
from torch_geometric.utils import to_dense_batch
from torch.nn import functional as F
from models.base import TGTransformerBaseModel, TGTransformerBaseModel_atom3d


class ProGSNN(TGTransformerBaseModel):

    def __init__(self, hparams):
        super(ProGSNN, self).__init__(hparams)

        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)

        self.save_hyperparameters()

        # model atributes
        self.input_dim = hparams.input_dim
        self.latent_dim = hparams.latent_dim
        self.hidden_dim = hparams.hidden_dim  # nhid
        self.embedding_dim = hparams.embedding_dim
        self.max_seq_len = hparams.prot_graph_size

        self.layers = hparams.layers
        self.probs = hparams.probs
        self.nhead = hparams.nhead
        self.src_mask = None
        self.lr = hparams.lr
        self.task = hparams.task
        self.alpha = hparams.alpha
        self.beta_loss = hparams.beta_loss
        self.batch_size = hparams.batch_size

        # Encoder
        self.scattering_network = Scatter(self.input_dim, self.max_seq_len, trainable_f=True)

        self.pos_encoder = PositionalEncoding(
            d_model=self.scattering_network.out_shape(),
            max_len=self.max_seq_len)

        self.row_encoder = TransformerEncoder(
            num_layers=self.layers,
            input_dim=self.scattering_network.out_shape(),
            num_heads=self.nhead,
            dim_feedforward=self.hidden_dim,
            dropout=self.probs)

        self.col_encoder = TransformerEncoder(num_layers=self.layers,
                                              input_dim=self.max_seq_len,
                                              num_heads=self.nhead,
                                              dim_feedforward=self.hidden_dim,
                                              dropout=self.probs)

        # Auxiliary network
        self.bottleneck_module = BaseBottleneck(
            self.scattering_network.out_shape(),
            self.latent_dim)

        # Property prediction
        # self regressor module
        proto_pred_net = str2auxnetwork(self.task)
        self.pred_net = proto_pred_net(hparams)

        #Can we use the same regressor module for time prediction as well?
       
        self.fc1 = nn.Linear(self.latent_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.scattering_network.out_shape())
        self.fc3 = nn.Linear(self.scattering_network.out_shape(), self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, 3)
        self.softmax = nn.Softmax(dim=0)
        self.loss_list = []


    def generate_row_mask(self, curr_seq_len):
        """create mask for transformer
        Args:
            max_seq_len (int): sequence length
        Returns:
            torch tensor : returns upper tri tensor of shape (S x S)
        """
        mask = torch.ones((self.max_seq_len, self.max_seq_len),
                          device=self.device)
        mask[:, curr_seq_len:] = 0.0
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
            mask == 1, float(0.0))
        return mask

    def row_transformer_encoding(self, embedded_batch):
        """transformer logic
        Args:
            embedded_batch ([type]): output of nn.Embedding layer (B x S X E)
        mask shape: S x S
        """
        #embedded_batch has shape [100,660,165]
        # print(embedded_batch.shape)
        pos_encoded_batch = self.pos_encoder(embedded_batch)
        # print(pos_encoded_batch.shape)
        # TransformerEncoder takes input (sequence_length,batch_size,num_features)
        row_mask = self.generate_row_mask(embedded_batch.shape[1])
        output_embed = self.row_encoder(pos_encoded_batch, row_mask)
        # output_embed = self.row_encoder(pos_encoded_batch, None)
        att_maps = self.row_encoder.get_attention_maps(pos_encoded_batch)
        return output_embed, att_maps

    def col_transformer_encoding(self, embedded_batch):
        """transformer logic
        Args:
            embedded_batch ([type]): output of nn.Embedding layer (B x S X E)
        mask shape: S x S
        """

        embedded_batch = embedded_batch.transpose(-1, -2)

        # TransformerEncoder takes input (sequence_length,batch_size,num_features)
        output_embed = self.col_encoder(embedded_batch, None)
        attention_maps = self.col_encoder.get_attention_maps(embedded_batch)

        return output_embed, attention_maps

    def reconstruct(self, z_rep):
        # Reconstruct the scattering coefficients.
        z_rep_expanded = z_rep.unsqueeze(1).repeat(1, self.max_seq_len, 1)
        h = F.relu(self.fc1(z_rep_expanded))
        return self.fc2(h)

    def reconstruct_coords(self, coeffs):
        #Reconstruct the x,y,z coordinates from the scattering coefficients

        h = F.relu(self.fc3(coeffs))
        return self.fc4(h)
    def encode(self, batch):
        """
        input data is a torch geometric mini-batch
        """

        # Scattering coefficients.
        #in_channels in the scattering network is 15 which corresponds to 15 amino acids
        #When we get the out_shape(), it is multiplied by 11 which equals 165.
        coeffs = self.scattering_network(batch)
        # print(coeffs)
        # print(coeffs.shape)
        #Scattering coefficients have shape [100,660,165] where 660 seems like the # of residues aka graph size, 165 is 11 times 15 where 15 is the # of AA.
        
        if len(coeffs.shape) == 2:
            coeffs = coeffs.unsqueeze(0)

        # print("Scattering completed!")
        #Row transformer encoding outputs attention map of shape [100,1,660,660]
        row_output_embed, att_maps = self.row_transformer_encoding(coeffs)
        # print("Row output embed shape")
        # print(row_output_embed.shape)
        #Col transformer encoding outputs attention map of shape [100,1,165,165]
        #We want to average over the 11 channels to get an attention map of [100,1,15,15]
        
        col_output_embed, attention_maps = self.col_transformer_encoding(coeffs)
        # print("col output embed shape")
        # print(col_output_embed.shape)
        output_embed = row_output_embed + col_output_embed.transpose(-1, -2)
        #z_rep as input. Predict time value and switch to DE Shaw dataloader.
        #Construct a feed forward network for this
        z_rep = output_embed.sum(1)
        # print("Latent representations shape")
        # print(z_rep.shape)

        # To regain the batch dimension
        if len(z_rep.shape) == 1:
            z_rep = z_rep.unsqueeze(0)

        z_rep = self.bottleneck_module(z_rep)

        # Reconstruct the scattering coefficients.
        coeffs_recon = self.reconstruct(z_rep)
        #Reconstruct the x,y,z coordinates from the scattering coefficients
        # print(coeffs_recon.shape)
        coords_recon = self.reconstruct_coords(coeffs_recon)
        return z_rep, coeffs, coeffs_recon, attention_maps, att_maps, coords_recon

    def forward(self, batch):
        # print(self.scattering_network.out_shape())
        z_rep, coeffs, coeffs_recon, attn_maps, att_maps, coords_recon = self.encode(batch)
        # print(attn_maps)
        #MLP for property prediction
        y_pred = self.pred_net(z_rep)
        # print(y_pred)
        # y_pred = self.softmax(y_pred)
        return y_pred, z_rep, coeffs_recon, coeffs, attn_maps, att_maps, coords_recon

    def main_loss(self, predictions, targets):
        y_pred = predictions
        y_true = targets

        # enrichment pred loss
        if self.task == 'reg':
            # import pdb; pdb.set_trace()
            #Comment out the two lines below when switching datasets
            y_pred = torch.tensor(y_pred, dtype=torch.float32)
            y_true = torch.tensor(y_true, dtype=torch.float32)
            loss = nn.MSELoss()(y_pred, y_true)
        elif self.task == 'bin_class':
            # import pdb; pdb.set_trace()
            loss = nn.BCEWithLogitsLoss()(y_pred.flatten(), y_true.flatten())
        elif self.task == 'multi_class':
            loss = nn.CrossEntropyLoss()(y_pred, y_true)
        # print("loss: {}".format(loss))
        return loss

    def recon_loss(self, predictions, targets):
        y_pred = predictions
        y_true = targets

        loss = nn.MSELoss()(y_pred.flatten(), y_true.flatten())
        # print("loss: {}".format(loss))
        return loss
    
    def recon_coords_loss(self, predictions, targets):
        y_pred = predictions
        y_true = targets
        print(y_pred.shape)
        print(y_true.shape)

        loss = nn.MSELoss()(y_pred.flatten(), y_true.flatten())
        # print("loss: {}".format(loss))
        return loss

    def get_loss_list(self):
        return self.loss_list
    
    # def training_step(self, batch, batch_idx):
    #     x, y  = batch
    #     x = x.float()
    #     y = y.float()

    #     y_pred, z_rep, coeffs_recon, coeffs, _, _ = self(x)
    #     recon_loss = self.recon_loss(coeffs_recon.flatten(), coeffs.flatten())
    #     main_loss = self.main_loss(y_pred, y) 


class ProGSNN_atom3d(TGTransformerBaseModel_atom3d):

    def __init__(self, hparams):
        super(ProGSNN, self).__init__(hparams)

        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)

        self.save_hyperparameters()

        # model atributes
        self.input_dim = hparams.input_dim
        self.latent_dim = hparams.latent_dim
        self.hidden_dim = hparams.hidden_dim  # nhid
        self.embedding_dim = hparams.embedding_dim
        self.max_seq_len = hparams.prot_graph_size

        self.layers = hparams.layers
        self.probs = hparams.probs
        self.nhead = hparams.nhead
        self.src_mask = None
        self.lr = hparams.lr
        self.task = hparams.task
        self.alpha = hparams.alpha
        self.beta_loss = hparams.beta_loss
        self.batch_size = hparams.batch_size

        # Encoder
        self.scattering_network = Scatter(self.input_dim, self.max_seq_len, trainable_f=True)

        self.pos_encoder = PositionalEncoding(
            d_model=self.scattering_network.out_shape(),
            max_len=self.max_seq_len)

        self.row_encoder = TransformerEncoder(
            num_layers=self.layers,
            input_dim=self.scattering_network.out_shape(),
            num_heads=self.nhead,
            dim_feedforward=self.hidden_dim,
            dropout=self.probs)

        self.col_encoder = TransformerEncoder(num_layers=self.layers,
                                              input_dim=self.max_seq_len,
                                              num_heads=self.nhead,
                                              dim_feedforward=self.hidden_dim,
                                              dropout=self.probs)

        # Auxiliary network
        self.bottleneck_module = BaseBottleneck(
            self.scattering_network.out_shape(),
            self.latent_dim)

        # Property prediction
        # self regressor module
        proto_pred_net = str2auxnetwork(self.task)
        self.pred_net = proto_pred_net(hparams)

        #Can we use the same regressor module for time prediction as well?
       
        self.fc1 = nn.Linear(self.latent_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.scattering_network.out_shape())
        self.fc3 = nn.Linear(self.scattering_network.out_shape(), self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, 3)
        self.softmax = nn.Softmax(dim=0)
        self.loss_list = []


    def generate_row_mask(self, curr_seq_len):
        """create mask for transformer
        Args:
            max_seq_len (int): sequence length
        Returns:
            torch tensor : returns upper tri tensor of shape (S x S)
        """
        mask = torch.ones((self.max_seq_len, self.max_seq_len),
                          device=self.device)
        mask[:, curr_seq_len:] = 0.0
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
            mask == 1, float(0.0))
        return mask

    def row_transformer_encoding(self, embedded_batch):
        """transformer logic
        Args:
            embedded_batch ([type]): output of nn.Embedding layer (B x S X E)
        mask shape: S x S
        """
        #embedded_batch has shape [100,660,165]
        # print(embedded_batch.shape)
        pos_encoded_batch = self.pos_encoder(embedded_batch)
        # print(pos_encoded_batch.shape)
        # TransformerEncoder takes input (sequence_length,batch_size,num_features)
        row_mask = self.generate_row_mask(embedded_batch.shape[1])
        output_embed = self.row_encoder(pos_encoded_batch, row_mask)
        # output_embed = self.row_encoder(pos_encoded_batch, None)
        att_maps = self.row_encoder.get_attention_maps(pos_encoded_batch)
        return output_embed, att_maps

    def col_transformer_encoding(self, embedded_batch):
        """transformer logic
        Args:
            embedded_batch ([type]): output of nn.Embedding layer (B x S X E)
        mask shape: S x S
        """

        embedded_batch = embedded_batch.transpose(-1, -2)

        # TransformerEncoder takes input (sequence_length,batch_size,num_features)
        output_embed = self.col_encoder(embedded_batch, None)
        attention_maps = self.col_encoder.get_attention_maps(embedded_batch)

        return output_embed, attention_maps

    def reconstruct(self, z_rep):
        # Reconstruct the scattering coefficients.
        z_rep_expanded = z_rep.unsqueeze(1).repeat(1, self.max_seq_len, 1)
        h = F.relu(self.fc1(z_rep_expanded))
        return self.fc2(h)

    # def reconstruct_coords(self, coeffs):
    #     #Reconstruct the x,y,z coordinates from the scattering coefficients

    #     h = F.relu(self.fc3(coeffs))
    #     return self.fc4(h)
    def encode(self, batch):
        """
        input data is a torch geometric mini-batch
        """

        # Scattering coefficients.
        #in_channels in the scattering network is 15 which corresponds to 15 amino acids
        #When we get the out_shape(), it is multiplied by 11 which equals 165.
        coeffs = self.scattering_network(batch)
        # print(coeffs)
        # print(coeffs.shape)
        #Scattering coefficients have shape [100,660,165] where 660 seems like the # of residues aka graph size, 165 is 11 times 15 where 15 is the # of AA.
        
        if len(coeffs.shape) == 2:
            coeffs = coeffs.unsqueeze(0)

        # print("Scattering completed!")
        #Row transformer encoding outputs attention map of shape [100,1,660,660]
        row_output_embed, att_maps = self.row_transformer_encoding(coeffs)
        # print("Row output embed shape")
        # print(row_output_embed.shape)
        #Col transformer encoding outputs attention map of shape [100,1,165,165]
        #We want to average over the 11 channels to get an attention map of [100,1,15,15]
        
        col_output_embed, attention_maps = self.col_transformer_encoding(coeffs)
        # print("col output embed shape")
        # print(col_output_embed.shape)
        output_embed = row_output_embed + col_output_embed.transpose(-1, -2)
        #z_rep as input. Predict time value and switch to DE Shaw dataloader.
        #Construct a feed forward network for this
        z_rep = output_embed.sum(1)
        # print("Latent representations shape")
        # print(z_rep.shape)

        # To regain the batch dimension
        if len(z_rep.shape) == 1:
            z_rep = z_rep.unsqueeze(0)

        z_rep = self.bottleneck_module(z_rep)

        # Reconstruct the scattering coefficients.
        coeffs_recon = self.reconstruct(z_rep)
        #Reconstruct the x,y,z coordinates from the scattering coefficients
        # print(coeffs_recon.shape)
        # coords_recon = self.reconstruct_coords(coeffs_recon)
        return z_rep, coeffs, coeffs_recon, attention_maps, att_maps

    def forward(self, batch):
        # print(self.scattering_network.out_shape())
        z_rep, coeffs, coeffs_recon, attn_maps, att_maps, coords_recon = self.encode(batch)
        # print(attn_maps)
        #MLP for property prediction
        y_pred = self.pred_net(z_rep)
        # print(y_pred)
        # y_pred = self.softmax(y_pred)
        return y_pred, z_rep, coeffs_recon, coeffs, attn_maps, att_maps

    def main_loss(self, predictions, targets):
        y_pred = predictions
        y_true = targets

        # enrichment pred loss
        if self.task == 'reg':
            # import pdb; pdb.set_trace()
            #Comment out the two lines below when switching datasets
            # y_pred = torch.tensor(y_pred, dtype=torch.float32)
            # y_true = torch.tensor(y_true, dtype=torch.float32)
            loss = nn.MSELoss()(y_pred, y_true)
        elif self.task == 'bin_class':
            # import pdb; pdb.set_trace()
            loss = nn.BCEWithLogitsLoss()(y_pred.flatten(), y_true.flatten())
        elif self.task == 'multi_class':
            loss = nn.CrossEntropyLoss()(y_pred, y_true)
        # print("loss: {}".format(loss))
        return loss

    def recon_loss(self, predictions, targets):
        y_pred = predictions
        y_true = targets

        loss = nn.MSELoss()(y_pred.flatten(), y_true.flatten())
        # print("loss: {}".format(loss))
        return loss
    

    def get_loss_list(self):
        return self.loss_list
    
        
    
    

    