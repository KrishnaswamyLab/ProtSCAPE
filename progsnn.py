import argparse

import torch
import torch.nn as nn
from modules.auxnetwork import str2auxnetwork
from modules.bottleneck import BaseBottleneck
from modules.scatter import Scatter
from modules.transformer import PositionalEncoding, TransformerEncoder
from torch_geometric.utils import to_dense_batch

from .base import TGTransformerBaseModel


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
        self.batch_size = hparams.batch_size

        # Encoder
        self.scattering_network = Scatter(in_channels=self.input_dim,
                                          skip_aggregation=True)

        self.pos_encoder = PositionalEncoding(
            d_model=self.scattering_network.out_shape_skip_aggregation(),
            max_len=self.max_seq_len)

        self.row_encoder = TransformerEncoder(
            num_layers=self.layers,
            input_dim=self.scattering_network.out_shape_skip_aggregation(),
            num_heads=self.nhead,
            dim_feedforward=self.hidden_dim,
            dropout=self.probs)

        self.col_encoder = TransformerEncoder(num_layers=self.layers,
                                              input_dim=self.max_seq_len,
                                              num_heads=2,
                                              dim_feedforward=self.hidden_dim,
                                              dropout=self.probs)

        # Auxiliary network
        self.bottleneck_module = BaseBottleneck(
            self.scattering_network.out_shape_skip_aggregation(),
            self.latent_dim)

        # Property prediction
        # self regressor module
        proto_pred_net = str2auxnetwork(self.task)
        self.pred_net = proto_pred_net(hparams)

        # Reconstruction of the scattering coefficients.
        # self.recon = nn.Sequential(
        #     nn.Linear(self.scattering_network.out_shape_skip_aggregation(), self.hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_dim, self.scattering_network.out_shape_skip_aggregation()))
        self.recon = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim,
                      self.scattering_network.out_shape_skip_aggregation()))

        self.loss_list = []

    def generate_zero_padding(self, dense_batch):
        graph_size = dense_batch.shape[1]
        padding_size = self.max_seq_len - graph_size
        padding = torch.zeros(
            (dense_batch.shape[0], padding_size, dense_batch.shape[-1]))
        return padding.type_as(dense_batch)

    def zero_pad_coeffs(self, raw_coeffs, batch):
        # Zero-pad the coefficients to the same dimension for the transformer modules.
        # [batch_size, max_node_size, coeff_dim].
        coeffs = to_dense_batch(raw_coeffs, batch.batch)[0]

        coeffs = coeffs.reshape(
            batch.num_graphs, -1,
            self.scattering_network.out_shape_skip_aggregation())
        padding = self.generate_zero_padding(coeffs)
        coeffs = torch.cat((coeffs, padding), dim=1)
        return coeffs

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
        pos_encoded_batch = self.pos_encoder(embedded_batch)

        # TransformerEncoder takes input (sequence_length,batch_size,num_features)
        # row_mask = self.generate_row_mask(embedded_batch.shape[1])
        # output_embed = self.row_encoder(pos_encoded_batch, row_mask)
        output_embed = self.row_encoder(pos_encoded_batch, None)

        return output_embed

    def col_transformer_encoding(self, embedded_batch):
        """transformer logic
        Args:
            embedded_batch ([type]): output of nn.Embedding layer (B x S X E)
        mask shape: S x S
        """

        embedded_batch = embedded_batch.transpose(-1, -2)

        # TransformerEncoder takes input (sequence_length,batch_size,num_features)
        output_embed = self.col_encoder(embedded_batch, None)

        return output_embed

    def encode(self, batch):
        """
        input data is a torch geometric mini-batch
        """

        # Scattering coefficients.
        raw_coeffs, _ = self.scattering_network(batch)
        # Zero-pad the coefficients to the same dimension for the transformer modules.
        coeffs = self.zero_pad_coeffs(raw_coeffs, batch)

        if len(coeffs.shape) == 2:
            coeffs = coeffs.unsqueeze(0)

        row_output_embed = self.row_transformer_encoding(coeffs)
        col_output_embed = self.col_transformer_encoding(coeffs)

        output_embed = row_output_embed + col_output_embed.transpose(-1, -2)

        z_rep = output_embed.sum(1)

        # To regain the batch dimension
        if len(z_rep.shape) == 1:
            z_rep = z_rep.unsqueeze(0)

        z_rep = self.bottleneck_module(z_rep)

        # Reconstruct the scattering coefficients.
        z_rep_expanded = z_rep.unsqueeze(1).repeat(1, self.max_seq_len, 1)
        coeffs_recon = self.recon(z_rep_expanded)

        return z_rep, coeffs, coeffs_recon

    def forward(self, batch):
        z_rep, coeffs, coeffs_recon = self.encode(batch)

        y_pred = self.pred_net(z_rep)

        return y_pred, z_rep, coeffs_recon, coeffs

    def main_loss(self, predictions, targets):
        y_pred = predictions
        y_true = targets

        # enrichment pred loss
        if self.task == 'reg':
            loss = nn.MSELoss()(y_pred.flatten(), y_true.flatten())
        elif self.task == 'bin_class':
            loss = nn.BCEWithLogitsLoss()(y_pred.flatten(), y_true.flatten())
        elif self.task == 'multi_class':
            loss = nn.CrossEntropyLoss()(y_pred, y_true)

        return loss

    def recon_loss(self, predictions, targets):
        y_pred = predictions
        y_true = targets

        loss = nn.MSELoss()(y_pred.flatten(), y_true.flatten())
        return loss

    def get_loss_list(self):
        return self.loss_list
