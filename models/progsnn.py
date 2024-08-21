import argparse

import torch
import torch.nn as nn
from models.auxnetwork import str2auxnetwork
from models.bottleneck_progsnn import BaseBottleneck
from models.scatter import Scatter
from models.transformer import PositionalEncoding, TransformerEncoder
from torch_geometric.utils import to_dense_batch
from torch.nn import functional as F
from models.base import TGTransformerBaseModel, TGTransformerBaseModel_atom3d, TGTransformerBaseModel_ATLAS, TGTransformerBaseModel_ATLAS_prop, TGTransformerBaseModel_ATLAS_noT
from models.scatter_deshaw import Scatter_deshaw
   
class ProGSNN_ATLAS(TGTransformerBaseModel_ATLAS):

    def __init__(self, hparams):
        super(ProGSNN_ATLAS, self).__init__(hparams)

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
        self.residue_num = hparams.residue_num
        self.protein = hparams.protein
        self.gamma = hparams.gamma
        # self.delta = hparams.delta

        # Encoder
        print("Initializing scattering..")
        self.scattering_network = Scatter_deshaw(self.input_dim, self.max_seq_len, trainable_f=True)
        print("Initializing positional encoding..")
        self.pos_encoder = PositionalEncoding(
            d_model=self.scattering_network.out_shape(),
            max_len=self.max_seq_len)
        print("Initializing row encoder..")
        self.row_encoder = TransformerEncoder(
            num_layers=self.layers,
            input_dim=self.scattering_network.out_shape(),
            num_heads=self.nhead,
            dim_feedforward=self.hidden_dim,
            dropout=self.probs)
        
        # import pdb; pdb.set_trace()
        print("Initializing col encoder..")
        self.col_encoder = TransformerEncoder(num_layers=self.layers,
                                              input_dim=self.residue_num,
                                              num_heads=self.nhead,
                                              dim_feedforward=self.hidden_dim,
                                              dropout=self.probs)

        # Auxiliary network
        print("Initializing bottleneck module..")
        self.bottleneck_module = BaseBottleneck(
            self.scattering_network.out_shape(),
            self.hidden_dim,
            self.latent_dim)

        # Property prediction
        # self regressor module
        print("Initializing prediction network..")
        proto_pred_net = str2auxnetwork(self.task)
        self.pred_net = proto_pred_net(hparams)
        self.node_encoder = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.node_decoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 20)
        )
        #Can we use the same regressor module for time prediction as well?
        # self.embedding = nn.Embedding(self.residue_num, 3)
        print("Initializing layer 1..")
        self.fc1 = nn.Linear(self.latent_dim, self.residue_num*self.hidden_dim)
        print("Initializing layer 2..")
        # import pdb; pdb.set_trace()
        self.fc2 = nn.Linear(self.residue_num*self.hidden_dim, self.residue_num*self.scattering_network.out_shape())
        print("Initializing layer 3..")
        self.fc3 = nn.Linear(self.latent_dim, self.hidden_dim)
        print("Initializing layer 4..")
        self.fc4 = nn.Linear(self.hidden_dim, 128)
        print("Initializing layer 5..")
        self.fc5 = nn.Linear(128, 64)
        print("Initializing layer 6..")
        self.fc6 = nn.Linear(64, 32)
        print("Initializing layer 7..")
        self.fc7 = nn.Linear(32, self.residue_num*self.residue_num*2)
        self.softmax = nn.Softmax(dim=0)
        self.loss_list = []
        print("Initialization complete!")


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
        #Create a vector of size sequence length and multiply it with the pos_encoded_batch
        #Ultimately we want (batch_size,seq_len*num_features) --> output_emved
        #vector --> att_maps
        # row_mask = self.generate_row_mask(embedded_batch.shape[1])
        output_embed = self.row_encoder(pos_encoded_batch)
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
        z_rep_expanded = z_rep.unsqueeze(1)
        # import pdb; pdb.set_trace()
        h = F.relu(self.fc1(z_rep_expanded))
        # import pdb; pdb.set_trace()
        return self.fc2(h)

    def reconstruct_coords(self, coeffs):
        #Reconstruct the x,y,z coordinates from the scattering coefficients

        h = F.relu(self.fc3(coeffs))
        h = F.relu(self.fc4(h))
        h = F.relu(self.fc5(h))
        h = F.relu(self.fc6(h))
        h = self.fc7(h)
        # random = h
        # import pdb; pdb.set_trace()
        # return h.reshape(-1, self.residue_num, 3)
        # return h.reshape(-1, self.residue_num, 2)
        h = h.reshape(-1, self.residue_num, self.residue_num, 2)
        return h

    def encode_nodes(self, batch):
        return self.node_encoder(batch.x)
    def decode_nodes(self, batch):
        return self.node_decoder(batch.x)
    def encode(self, batch):
        """
        input data is a torch geometric mini-batch
        """
        aa_gt = batch.x
        batch.x = self.encode_nodes(batch)
        # Scattering coefficients.
        
        # import pdb; pdb.set_trace()
        print("Before Scattering")
        #in_channels in the scattering network is 15 which corresponds to 15 amino acids
        #When we get the out_shape(), it is multiplied by 11 which equals 165.
        coeffs = self.scattering_network(batch)
        # print(coeffs)
        # print(coeffs.shape)
        #Scattering coefficients have shape [100,204,165] where 204 seems like the # of residues aka graph size, 165 is 11 times 15 where 15 is the # of AA.
        
        if len(coeffs.shape) == 2:
            coeffs = coeffs.unsqueeze(0)
        # import pdb; pdb.set_trace()
        print("Scattering completed!")
        #Row transformer encoding outputs attention map of shape [100,1,204,204]
        row_output_embed, att_maps = self.row_transformer_encoding(coeffs)
        # print("Row output embed shape")
        # print(row_output_embed.shape)
        #Col transformer encoding outputs attention map of shape [100,1,165,165]
        #We want to average over the 11 channels to get an attention map of [100,1,15,15]
        # import pdb; pdb.set_trace() 
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
        coeffs_recon = coeffs_recon.reshape(-1, self.residue_num, self.scattering_network.out_shape())

        #Reconstruct the one hot encoding of the amino acids
        aa_recon = self.decode_nodes(batch)
        return z_rep, coeffs, coeffs_recon, attention_maps, att_maps, aa_recon, aa_gt

    def forward(self, batch):
        # print(self.scattering_network.out_shape())
        z_rep, coeffs, coeffs_recon, attn_maps, att_maps, aa_recon, aa_gt = self.encode(batch)
        # print(attn_maps)
        #MLP for property prediction
        y_pred = self.pred_net(z_rep)
        coords_recon = self.reconstruct_coords(z_rep)
        # import pdb; pdb.set_trace()
        # coords_recon = coords_recon.reshape(-1, self.residue_num, 3)
        # print(y_pred)
        # import pdb; pdb.set_trace()
        # y_pred = self.softmax(y_pred)
        return y_pred, z_rep, coeffs_recon, coeffs, attn_maps, att_maps, coords_recon, aa_recon, aa_gt

    def main_loss(self, predictions, targets):
        y_pred = predictions
        y_true = targets
        # import pdb; pdb.set_trace()
        # enrichment pred loss
        if self.task == 'reg':
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # import pdb; pdb.set_trace()
            #Comment out the two lines below when switching datasets
            # y_pred = torch.tensor(y_pred, dtype=torch.float32, requires_grad=True)
            y_true = torch.tensor(y_true, dtype=torch.float32, requires_grad=True).to(device)
            if y_true.shape != y_pred.shape:
                # assert y_true.shape[0] == y_pred.shape[0]
                y_true = y_true.unsqueeze(-1)
            loss = nn.MSELoss()(y_pred, y_true)
        elif self.task == 'bin_class':
            # import pdb; pdb.set_trace()
            loss = nn.BCEWithLogitsLoss()(y_pred.flatten(), y_true.flatten())
        elif self.task == 'multi_class':
            loss = nn.CrossEntropyLoss()(y_pred, y_true)
        # print("loss: {}".format(loss))
        return loss

    def recon_loss(self, predictions, targets):
        y_pred = predictions.float()
        y_true = targets.float()
        # y_pred = torch.tensor(y_pred, dtype=torch.float32)
        # y_true = torch.tensor(y_true, dtype=torch.float32)
        # import pdb; pdb.set_trace()
        loss = nn.MSELoss()(y_pred, y_true)
        # print("loss: {}".format(loss))
        return loss
    
    def recon_aa_loss(self, predictions, targets):
        y_pred = predictions.float()
        y_true = targets.float()
        # y_pred = torch.tensor(y_pred, dtype=torch.float32)
        # y_true = torch.tensor(y_true, dtype=torch.float32)
        # import pdb; pdb.set_trace()
        loss = nn.MSELoss()(y_pred, y_true)
        # print("loss: {}".format(loss))
        return loss
    
    def recon_coords_loss(self, predictions, targets):
        # import pdb; pdb.set_trace()
        y_pred = predictions
        # import pdb; pdb.set_trace()
        # y_true = torch.stack(targets)
        # y_pred = torch.tensor(y_pred, dtype=torch.float32)
        y_true = torch.tensor(targets, dtype=torch.float32, device='cuda')
        if self.protein == 'gb3' or self.protein == 'bpti':
            y_true = y_true.reshape(-1, self.residue_num, 3)
        # print(y_pred.shape)
        # print(y_true.shape)
        # import pdb; pdb.set_trace()
        
        # if y_true.shape[0] == 5600:
        #     y_true = y_true[:5600 - (5600 % 100), :].reshape(100, -1, 3)
        # elif y_true.shape[0] == 4928:
        #     y_true = y_true[:4928 - (4928 % 88), :].reshape(88, -1, 3) 
        # else:
        #     y_true = y_true[:5432 - (5432 % 97), :].reshape(97, -1, 3) 
        # import pdb; pdb.set_trace()
        # print(y_pred.shape)
        # print(y_true.shape)
        loss = nn.MSELoss()(y_pred, y_true)
        print("loss: {}".format(loss))
        return loss

    def kl_divergence(self, mu, logvar):
        # import pdb; pdb.set_trace()
        # mu = torch.tensor(mu, dtype=torch.float32)
        # logvar = torch.tensor(logvar, dtype=torch.float32)
        # import pdb; pdb.set_trace()
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_div

    def get_loss_list(self):
        return self.loss_list
    
    # def training_step(self, batch, batch_idx):
    #     x, y  = batch
    #     x = x.float()
    #     y = y.float()

    #     y_pred, z_rep, coeffs_recon, coeffs, _, _ = self(x)
    #     recon_loss = self.recon_loss(coeffs_recon.flatten(), coeffs.flatten())
    #     main_loss = self.main_loss(y_pred, y)

