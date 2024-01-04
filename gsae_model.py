import numpy as np

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

from pytorch_lightning.loggers import TensorBoardLogger

import pytorch_lightning as pl

class GSAE(pl.LightningModule):
    def __init__(self, input_dim, bottle_dim, hidden_dim,\
                    learning_rate, alpha, beta, n_epochs,\
                    len_epoch, batch_size, n_gpus, save_dir):
        super(GSAE, self).__init__()
        
        #self.hparams = hparams
        
        self.input_dim = input_dim
        self.bottle_dim = bottle_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.beta = beta
        self.n_epochs = n_epochs
        self.len_epoch = len_epoch
        self.validation_step_outputs = []


        self.fc11 = nn.Linear(self.input_dim, self.hidden_dim)
        self.bn11 = nn.BatchNorm1d(self.hidden_dim)
        
        self.fc12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.bn12 = nn.BatchNorm1d(self.hidden_dim)
        
        self.fc21 = nn.Linear(self.hidden_dim, self.bottle_dim)
        self.fc22 = nn.Linear(self.hidden_dim, self.bottle_dim)

        self.fc3 = nn.Linear(self.bottle_dim, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, self.input_dim)
        # energy prediction
        self.regfc1 = nn.Linear(self.bottle_dim, 20)
        self.regfc2 = nn.Linear(20, 1)

        self.loss_list = []

        
        if n_gpus > 0:
            self.dev_type = 'cuda'

        if n_gpus == 0:
            self.dev_type = 'cpu'
        
        self.eps = 1e-5


    def kl_div(self,mu, logvar):
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        
        return KLD
        
    # main model functions
    def encode(self, x):
        h = self.bn11(F.relu(self.fc11(x)))
        h = self.bn12(F.relu(self.fc12(h)))
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)
    
    def embed(self, x):
        # print(x.shape)
        h = self.bn11(F.relu(self.fc11(x)))
        h = self.bn12(F.relu(self.fc12(h)))
        mu = self.fc21(h)
        logvar = self.fc22(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar 

    def predict(self,z):
        h = F.relu(self.regfc1(z))
        y_pred = self.regfc2(h)
        return y_pred
    
    def predict_from_data(self,x):
        z = self.embed(x)[0]
        pred = self.predict(z)
        return pred

    def forward(self, x):
        # encoding
        z, mu, logvar = self.embed(x)
        # predict
        y_pred = self.predict(z)
        # recon
        x_hat = self.decode(z)

        return x_hat, y_pred, mu, logvar, z

    def loss_multi_GSAE(self, 
                        recon_x, x,  
                        mu, logvar,
                        y_pred, y, 
                        alpha, beta, batch_idx):

        # reconstruction loss
        recon_loss = nn.MSELoss()(recon_x.flatten(), x.flatten()) 
        
        # regression loss
        reg_loss = nn.MSELoss()(y_pred, y) 

        # kl divergence 
        KLD = self.kl_div(mu, logvar)

        num_epochs = self.n_epochs - 5
        total_batches = self.len_epoch * num_epochs

        # loss annealing
        weight = min(1, float(self.trainer.global_step) / float(total_batches))
        #reg_loss = weight * reg_loss
        kl_loss = weight* KLD
        
        reg_loss = alpha * reg_loss.mean()

        kl_loss = beta * kl_loss

        #total_loss = recon_loss + reg_loss + kl_loss
        total_loss = recon_loss + reg_loss

        #no regression
        #total_loss = recon_loss + kl_loss

        self.loss_list.append(total_loss.item())

        log_losses = {'train_loss' : total_loss.detach(), 
                    'recon_loss' : recon_loss.detach(),
                    'pred_loss' :reg_loss.detach(),
                    #'kl_loss': kl_loss.detach()}
        }
        
        return total_loss, log_losses

    def get_loss_list(self):
        return self.loss_list

    def training_step(self, batch, batch_idx):
        x, y  = batch
        x = x.float()
        y = y.float()
        x_hat, y_hat, mu, logvar, z = self(x)

        loss, log_losses = self.loss_multi_GSAE(recon_x=x, x=x_hat, 
                                                mu=mu, logvar=logvar,
                                                y_pred=y_hat, y=y,
                                            alpha=self.alpha, beta=self.beta,
                                            batch_idx=batch_idx)
            
        return {'loss': loss, 'log': log_losses}
   
    def validation_step(self, batch, batch_idx):
        x, y  = batch
        x = x.float()
        # print(x.shape)
        x_hat, y_hat, mu, logvar,z = self(x)

        # reconstruction loss
        recon_loss = nn.MSELoss()(x_hat.flatten(), x.flatten()) 
        
        # regression loss
        reg_loss = nn.MSELoss()(y_hat.reshape(-1), y.reshape(-1)) 

        # kl loss
        kl_loss = self.kl_div(mu, logvar)
    
        #total_loss = recon_loss  +  reg_loss + kl_loss
        total_loss = recon_loss  +  reg_loss
        #total_loss = recon_loss + kl_loss

        log_losses = {'val_loss' : total_loss.detach(), 
                    'val_recon_loss' : recon_loss.detach(),
                    'val_pred_loss' :reg_loss.detach(),
                    #'val_kl_loss': kl_loss.detach()}}
        }
        self.validation_step_outputs.append(log_losses)
        return log_losses

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_reconloss = torch.stack([x['val_recon_loss'] for x in outputs]).mean()
        avg_regloss = torch.stack([x['val_pred_loss'] for x in outputs]).mean()
        #avg_klloss = torch.stack([x['val_kl_loss'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss,
                            'val_avg_recon_loss': avg_reconloss,
                            'val_avg_pred_loss':avg_regloss,
                            #'val_avg_kl_loss':avg_klloss}
        }

        return {'val_loss': avg_loss, 'log': tensorboard_logs}


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

