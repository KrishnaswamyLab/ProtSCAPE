"""
Functions and classes for a variational auto-encoder (VAE).
"""

from metrics.metrics_fns import (
    convert_batch_coords,
    EuclideanDistanceCorrs
)
import utilities as U
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim


"""
FUNCTION: VAE LOSS
"""
def vae_loss(input_dict, 
             output_dict,
             target_name = 'target',
             rescale_wts_relatively = False,
             KLD_wt = 0.5):

    # parse input and output dictionaries
    x = input_dict['target'][target_name]
    x_hat, mean, logvar = (output_dict['x_hat'], 
                           output_dict['means'],
                           output_dict['logvars'])
    
    # losses
    KLD_loss = -torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) # originally -0.5 coeff.
    reconst_info_loss = F.mse_loss(x_hat, x, reduction='sum')

    if rescale_wts_relatively:
        KLD_loss_scaled = KLD_loss
        if (KLD_loss > reconst_info_loss):
            KLD_loss_scaled = KLD_wt * KLD_loss * (reconst_info_loss / KLD_loss)
    else:
        KLD_loss_scaled = KLD_wt * KLD_loss
        
    loss_dict = {
        'loss': reconst_info_loss + KLD_loss_scaled,
        'reconst_info_loss': reconst_info_loss,
        'KLD_loss_scaled': KLD_loss_scaled,
        'size': x.shape[0] # batch size
    }
        
    return loss_dict
    

"""
CLASS: VAE MODEL
"""
class VAE(nn.Module):
    def __init__(self,
                 loss_fn,
                 loss_fn_kwargs,
                 input_dim, 
                 output_dim,
                 encoder_dim_arr,
                 latent_dims,
                 decoder_dim_arr,
                 nonlin_fn,
                 nonlin_fn_kwargs,
                 wt_init_fn=nn.init.kaiming_uniform_,
                 wt_init_fn_kwargs={'nonlinearity': 'relu'},
                 decoder_final_nonlin_fn=nn.Sigmoid,
                 key_prefix='vae',
                 preds_targets_start_i=0):
        
        super(VAE, self).__init__()
        self.loss_fn = loss_fn
        self.loss_fn_kwargs = loss_fn_kwargs
        self.wt_init_fn = wt_init_fn
        self.wt_init_fn_kwargs = wt_init_fn_kwargs
        self.target_name = loss_fn_kwargs['target_name']
        self.preds_targets_start_i = preds_targets_start_i
        self.valid_metric_collection = None
        # track correlations for valid set during training
        self.EDCorrs = EuclideanDistanceCorrs(
            square_dists=False, 
            reduction='mean'
        )
        self.loss_keys = None # set once in update_metrics()
        self.epoch_loss_dict = {
            'train': {'size': 0.0}, 
            'valid': {'size': 0.0}
        }
        self.nonlin_fn = nonlin_fn
        self.nonlin_fn_kwargs = nonlin_fn_kwargs
        self.key_prefix = key_prefix
        
        # encoder pieces
        (self.encoder_lin_fns, 
         self.encoder_nonlin_fns, 
         self.encoder_lin_out) = U.build_ffnn(input_dim=input_dim,
                                              output_dim=latent_dims[0], 
                                              hidden_dim_array=encoder_dim_arr, 
                                              nonlin_fn=nonlin_fn,
                                              nonlin_fn_kwargs=nonlin_fn_kwargs)
        
        # latent space: parameterized by (multivariate) mean and 
        # log-variance layers
        self.ls_mean_layer = nn.Linear(latent_dims[0], latent_dims[1])
        self.ls_logvar_layer = nn.Linear(latent_dims[0], latent_dims[1])
        
        # decoder pieces
        self.decoder_final_nonlin_fn = decoder_final_nonlin_fn
        (self.decoder_lin_fns, 
         self.decoder_nonlin_fns, 
         self.decoder_lin_out) = U.build_ffnn(input_dim=latent_dims[1],
                                              output_dim=output_dim, 
                                              hidden_dim_array=decoder_dim_arr, 
                                              nonlin_fn=nonlin_fn,
                                              nonlin_fn_kwargs=nonlin_fn_kwargs)


    def __post_init__(self):
        # initialize linear layer weights
        lin_fns = self.encoder_lin_fns + \
                    self.encoder_lin_out + \
                    self.decoder_lin_fns
        for lin_fn in lin_fns:
            self._weights_init(lin_fn)
    
    
    def _weights_init(m):
        # random weights initialization for linear layers
        if isinstance(m, nn.Linear):
            self.wt_init_fn(m.weight.data, **self.wt_init_fn_kwargs)
            nn.init.zeros_(m.bias.data)

    
    def encode(self, x):
        for i in range(len(self.encoder_lin_fns)):
            x = self.encoder_lin_fns[i](x)
            x = self.encoder_nonlin_fns[i](x)
        x = self.encoder_lin_out(x)
        enc_final_nonlin_fn = self.nonlin_fn(**self.nonlin_fn_kwargs)
        x = enc_final_nonlin_fn(x)
        means, logvars = self.ls_mean_layer(x), self.ls_logvar_layer(x)
        return means, logvars

    
    def reparameterize(self, means, vars):
        epsilons = torch.randn_like(vars) # .to(device)
        zs = means + vars * epsilons
        return zs

    
    def decode(self, u):
        for i in range(len(self.decoder_lin_fns)):
            u = self.decoder_lin_fns[i](u)
            u = self.decoder_nonlin_fns[i](u)
        u = self.decoder_lin_out(u)
        decoder_out = self.decoder_final_nonlin_fn()
        u = decoder_out(u)
        return u

    
    def forward(self, x):
        if isinstance(x, dict):
            x = x['x']
        means, logvars = self.encode(x)
        if self.training:
            zs = self.reparameterize(means, logvars)
            x_hat = self.decode(zs)
        else:
            x_hat = self.decode(means)
        model_output_dict = {
            'x_hat': x_hat,
            'means': means,
            'logvars': logvars
        }
        return model_output_dict


    def loss(self, input_dict, output_dict):
        """
        This fn wraps loss_fn so it takes dicts storing 
        preds, targets as inputs, and outputs a loss_dict.
        """
        loss_dict = self.loss_fn(input_dict, 
                                 output_dict,
                                 **self.loss_fn_kwargs)
        return loss_dict

    
    def update_metrics(self, 
                       phase,
                       loss_dict,
                       input_dict, 
                       output_dict):

        # set loss_keys on first encounter
        if self.loss_keys is None:
            self.loss_keys = [k for k in loss_dict.keys() if 'loss' in k.lower()]
            for phase_name in self.epoch_loss_dict.keys():
                for loss_key in self.loss_keys:
                    self.epoch_loss_dict[phase_name][loss_key] = 0.0
                    
        # for both train and valid phases, update losses and sizes
        for loss_key in self.loss_keys:
            self.epoch_loss_dict[phase][loss_key] += loss_dict[loss_key].item()
        self.epoch_loss_dict[phase]['size'] += loss_dict['size'] 

        # validation metrics
        if phase == 'valid':
            preds = output_dict['x_hat']
            target = input_dict['target'][self.target_name]

            # calc distances PCC for each reconstructed conformation coords in the batch
            pred_coords = convert_batch_coords(preds, start_i=self.preds_targets_start_i)
            target_coords = convert_batch_coords(target, start_i=self.preds_targets_start_i)
            self.EDCorrs.update(pred_coords.cpu().numpy(), target_coords.cpu().numpy())

            # flatten tensors along first (batch_size) dimension
            preds = preds.reshape(1, -1).squeeze()
            target = target.reshape(1, -1).squeeze()
            if self.valid_metric_collection is not None:
                self.valid_metric_collection.update(preds, target)


    def calc_metrics(self, 
                     epoch,
                     input_dict = None):

        metrics_dict = {'epoch': epoch}

        # calc mean PCC of original and reconstruction coord distances
        corrs_dict = self.EDCorrs.compute()
        metrics_dict['mean_pcc_valid'] = corrs_dict['pcc']
        metrics_dict['mean_scc_valid'] = corrs_dict['scc']
        self.EDCorrs.reset()
        
        # calc validation metrics in MetricCollection (dictionary)
        if self.valid_metric_collection is not None:
            vmetrics = self.valid_metric_collection.compute()
            self.valid_metric_collection.reset()
            
            # save scalar metrics in metrics_dict, with epoch
            metrics_dict = metrics_dict \
                            | {(self.key_prefix + '_' + k + '_valid'): \
                                   v.detach().cpu().numpy().item() \
                               for (k, v) in vmetrics.items()}
        
        # include train and validation mean losses in metrics_dict
        for phase in ('train', 'valid'):
            for loss_key in self.loss_keys:
                avg_loss = self.epoch_loss_dict[phase][loss_key] \
                            / self.epoch_loss_dict[phase]['size']
                metrics_dict = metrics_dict | {(loss_key + '_' + phase): avg_loss}
                self.epoch_loss_dict[phase][loss_key] = 0.0
            self.epoch_loss_dict[phase]['size'] = 0.0
        
        return metrics_dict


def load_VAE_with_accelerate(args, 
                             vae_model_path, 
                             vae_dataloaders,
                             sets):
    """
    Loads a (trained) VAE state using accelerate.
    Returns the accelerate object, vae, and dataloaders
    for the specified sets (train/valid/test).
    """
    from accelerate import Accelerator
    
    # init new VAE
    vae = VAE(
        loss_fn=vae_loss,
        loss_fn_kwargs=None,
        input_dim=args.VAE_INPUT_DIM,
        encoder_dim_arr=args.ENCODER_DIM_ARR,
        latent_dims=args.LATENT_SPACE_DIMS,
        decoder_dim_arr=args.DECODER_DIM_ARR,
        nonlin_fn=nn.LeakyReLU,
        nonlin_fn_kwargs={'negative_slope': args.RELU_NSLOPE},
        wt_init_fn=nn.init.kaiming_uniform_,
        decoder_final_nonlin_fn=nn.Sigmoid
    )
    
    # must be the same optimizer used to train the vae!
    # TODO: this could be abstracted to accept an optimizer obj
    # as an arg...
    vae_optimizer = optim.Adam(
        vae.parameters(),
        lr=args.LEARN_RATE,
        betas=args.ADAM_BETAS
    )

    # a saved/trained VAE is wrapped by accelerate, so
    # we have to use accelerate's `load_state` to access
    acc = Accelerator(cpu=args.ON_CPU)
    (vae, vae_optimizer) = acc.prepare(vae, vae_optimizer)
    acc_vae_dataloaders = {set: acc.prepare(vae_dataloaders[set]) for set in sets}
    epoch_ctr = U.EpochCounter(0, args.MAIN_METRIC)
    acc.register_for_checkpointing(epoch_ctr)
    
    # finally, load trained VAE from save path
    acc.load_state(vae_model_path)

    return (acc, vae, acc_vae_dataloaders)

