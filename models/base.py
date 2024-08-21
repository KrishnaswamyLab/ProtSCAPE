import argparse
import sys

import numpy as np
import torch
from pytorch_lightning.core.module import LightningModule

sys.path.append('../')
from models.scheduler import CosineWarmupScheduler


class TGTransformerBaseModel(LightningModule):
    """
    base model for models which take in
    torch geometric batches

    """

    def __init__(self, hparams=None):
        super(TGTransformerBaseModel, self).__init__()

        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)

        self.save_hyperparameters()

        #self.hparams = hparams

        self.lr = hparams.lr
        self.task = hparams.task

    def configure_optimizers(self):
        # print("Going here")
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)

        lr_sched = CosineWarmupScheduler(opt, warmup=5000, max_iters=50000)

        return [opt], [lr_sched]

    def shared_step(self, batch):
        # print(batch)
        # import pdb; pdb.set_trace() 
        x, y = batch.x, batch.time
        coords = batch.coords
        # import pdb; pdb.set_trace()
        x = x.float()#, y.float()

        preds, _, coeffs_recon, coeffs, _,_, coords_recon = self(batch)
        targets = y

        return preds, targets, coeffs_recon, coeffs, coords, coords_recon

    def relabel(self, loss_dict, label):

        loss_dict = {label + str(key): val for key, val in loss_dict.items()}

        return loss_dict

    def training_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        preds, targets, coeffs_recon, coeffs, coords, coords_recon = self.shared_step(batch)

        assert len(preds) == len(
            targets), f'preds: {len(preds)} targs: {len(targets)}'

        train_loss, train_loss_logs = self.multi_loss(
            predictions=preds, targets=targets, coeffs_recon=coeffs_recon, coeffs=coeffs, batch_idx=batch_idx,
              coords=coords, coords_recon=coords_recon)

        train_loss_logs = self.relabel(train_loss_logs, 'train_')

        self.log_dict(train_loss_logs, on_step=True, on_epoch=True, batch_size=self.batch_size)

        return train_loss

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        preds, targets, coeffs_recon, coeffs, coords, coords_recon = self.shared_step(batch)
        # print(targets)
        assert len(preds) == len(
            targets), f'preds: {len(preds)} targs: {len(targets)}'

        val_loss, val_loss_logs = self.multi_loss(
            predictions=preds, targets=targets, coeffs_recon=coeffs_recon, coeffs=coeffs, batch_idx=batch_idx,
              coords=coords, coords_recon=coords_recon)

        val_loss_logs = self.relabel(val_loss_logs, 'val_')

        self.log_dict(val_loss_logs, on_step=True, on_epoch=True, batch_size=self.batch_size)

        return val_loss

    def multi_loss(self, predictions, targets, coeffs_recon, coeffs, batch_idx, coords, coords_recon):
        # main loss
        main_loss = self.main_loss(predictions=predictions, targets=targets)

        # reconstruction loss
        recon_loss = self.recon_loss(predictions=coeffs_recon, targets=coeffs)

        #reconstruction of coordinates loss
        # recon_coords_loss = self.recon_coords_loss(predictions=coords_recon, targets=coords)

        # total_loss = self.alpha * main_loss + self.beta_loss * recon_loss + (1-self.alpha-self.beta_loss) * recon_coords_loss
        total_loss = self.alpha*main_loss + recon_loss #+ recon_coords_loss

        log_losses = {
            'loss': total_loss.detach(),
            'main_loss': main_loss.detach(),
            'recon_loss': recon_loss.detach(),
            #'recon_coords_loss': recon_coords_loss.detach(),
        }

        return total_loss, log_losses

    def main_loss(self, predictions, targets, valid_step=False):
        """
        takes in predictions and targets

        predictions and targets can be a list

        Args:
            predictions (dict): [description]
            targets (dict): [description]

        Returns:
            loss value
        """
        raise NotImplementedError

    def recon_loss(self, predictions, targets, valid_step=False):
        """
        takes in predictions and targets

        predictions and targets can be a list

        Args:
            predictions (dict): [description]
            targets (dict): [description]

        Returns:
            loss value
        """
        raise NotImplementedError

    def forward(self, batch):
        """
        the forward method should just take data
        as an argument


        should return:
            -  predictions
            -  embeddings
        """
        return NotImplementedError

class TGTransformerBaseModel_atom3d(LightningModule):
    """
    base model for models which take in
    torch geometric batches

    """

    def __init__(self, hparams=None):
        super(TGTransformerBaseModel_atom3d, self).__init__()

        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)

        self.save_hyperparameters()

        #self.hparams = hparams

        self.lr = hparams.lr
        self.task = hparams.task

    def configure_optimizers(self):
        # print("Going here")
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)

        lr_sched = CosineWarmupScheduler(opt, warmup=5000, max_iters=50000)

        return [opt], [lr_sched]

    def shared_step(self, batch):
        # print(batch)
        #Change to batch.time when using DeShaw dataset
        x, y = batch.x, batch.y
        coords = batch.pos
        # import pdb; pdb.set_trace()
        x = x.float()#, y.float()

        preds, _, coeffs_recon, coeffs, _,_ = self(batch)
        targets = y

        return preds, targets, coeffs_recon, coeffs, coords

    def relabel(self, loss_dict, label):

        loss_dict = {label + str(key): val for key, val in loss_dict.items()}

        return loss_dict

    def training_step(self, batch, batch_idx):

        preds, targets, coeffs_recon, coeffs, coords = self.shared_step(batch)

        assert len(preds) == len(
            targets), f'preds: {len(preds)} targs: {len(targets)}'

        train_loss, train_loss_logs = self.multi_loss(
            predictions=preds, targets=targets, coeffs_recon=coeffs_recon, 
            coeffs=coeffs, batch_idx=batch_idx)

        train_loss_logs = self.relabel(train_loss_logs, 'train_')

        self.log_dict(train_loss_logs, on_step=True, on_epoch=True, batch_size=self.batch_size)

        return train_loss

    def validation_step(self, batch, batch_idx):

        preds, targets, coeffs_recon, coeffs, coords, coords_recon = self.shared_step(batch)
        # print(targets)
        assert len(preds) == len(
            targets), f'preds: {len(preds)} targs: {len(targets)}'

        val_loss, val_loss_logs = self.multi_loss(
            predictions=preds, targets=targets, coeffs_recon=coeffs_recon,
              coeffs=coeffs, batch_idx=batch_idx)

        val_loss_logs = self.relabel(val_loss_logs, 'val_')

        self.log_dict(val_loss_logs, on_step=True, on_epoch=True, batch_size=self.batch_size)

        return val_loss

    def multi_loss(self, predictions, targets, coeffs_recon, coeffs, batch_idx):
        # main loss
        main_loss = self.main_loss(predictions=predictions, targets=targets)

        # reconstruction loss
        recon_loss = self.recon_loss(predictions=coeffs_recon, targets=coeffs)

        #reconstruction of coordinates loss
        # recon_coords_loss = self.recon_coords_loss(predictions=coords_recon, targets=coords)

        total_loss =  main_loss + self.alpha *recon_loss
        # total_loss = main_loss

        log_losses = {
            'loss': total_loss.detach(),
            'main_loss': main_loss.detach(),
            'recon_loss': recon_loss.detach(),
        }

        return total_loss, log_losses

    def main_loss(self, predictions, targets, valid_step=False):
        """
        takes in predictions and targets

        predictions and targets can be a list

        Args:
            predictions (dict): [description]
            targets (dict): [description]

        Returns:
            loss value
        """
        raise NotImplementedError

    def recon_loss(self, predictions, targets, valid_step=False):
        """
        takes in predictions and targets

        predictions and targets can be a list

        Args:
            predictions (dict): [description]
            targets (dict): [description]

        Returns:
            loss value
        """
        raise NotImplementedError

    def forward(self, batch):
        """
        the forward method should just take data
        as an argument


        should return:
            -  predictions
            -  embeddings
        """
        return NotImplementedError

class TGTransformerBaseModel_ATLAS(LightningModule):
    """
    base model for models which take in
    torch geometric batches

    """

    def __init__(self, hparams=None):
        super(TGTransformerBaseModel_ATLAS, self).__init__()

        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)

        self.save_hyperparameters()

        #self.hparams = hparams

        self.lr = hparams.lr
        self.task = hparams.task

    def configure_optimizers(self):
        # print("Going here")
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)

        lr_sched = CosineWarmupScheduler(opt, warmup=5000, max_iters=50000)

        return [opt], [lr_sched]

    def shared_step(self, batch):
        # print(batch)
        # import pdb; pdb.set_trace() 
        x, y = batch.x, batch.time
        coords = batch.coords
        # import pdb; pdb.set_trace()
        coords = coords.view(batch.y.shape[0], coords.shape[1],coords.shape[1],2)
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        x = x.float()#, y.float()

        preds, _, coeffs_recon, coeffs, _,_, coords_recon, aa_recon, aa_gt = self(batch)
        targets = y

        return preds, targets, coeffs_recon, coeffs, coords, coords_recon, aa_recon, aa_gt

    def relabel(self, loss_dict, label):

        loss_dict = {label + str(key): val for key, val in loss_dict.items()}

        return loss_dict

    def training_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        preds, targets, coeffs_recon, coeffs, coords, coords_recon, aa_recon, aa_gt, = self.shared_step(batch)

        assert len(preds) == len(
            targets), f'preds: {len(preds)} targs: {len(targets)}'

        train_loss, train_loss_logs = self.multi_loss(
            predictions=preds, targets=targets, coeffs_recon=coeffs_recon, coeffs=coeffs, batch_idx=batch_idx,
              coords=coords, coords_recon=coords_recon, aa_recon=aa_recon, aa_gt=aa_gt)

        train_loss_logs = self.relabel(train_loss_logs, 'train_')

        self.log_dict(train_loss_logs, on_step=False, on_epoch=True, batch_size=self.batch_size)

        return train_loss

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        preds, targets, coeffs_recon, coeffs, coords, coords_recon, aa_recon, aa_gt = self.shared_step(batch)
        # print(targets)
        assert len(preds) == len(
            targets), f'preds: {len(preds)} targs: {len(targets)}'

        val_loss, val_loss_logs = self.multi_loss(
            predictions=preds, targets=targets, coeffs_recon=coeffs_recon, coeffs=coeffs, batch_idx=batch_idx,
              coords=coords, coords_recon=coords_recon, aa_recon=aa_recon, aa_gt=aa_gt)

        val_loss_logs = self.relabel(val_loss_logs, 'val_')

        self.log_dict(val_loss_logs, on_step=False, on_epoch=True, batch_size=self.batch_size)

        return val_loss

    def multi_loss(self, predictions, targets, coeffs_recon, coeffs, batch_idx, coords, coords_recon, aa_recon, aa_gt):
        # main loss
        # main_loss = self.main_loss(predictions=predictions, targets=targets)

        # reconstruction loss
        recon_loss = self.recon_loss(predictions=coeffs_recon, targets=coeffs)

        #reconstruction of coordinates loss
        recon_coords_loss = self.recon_coords_loss(predictions=coords_recon, targets=coords)
        print("Done printing recon loss")
        #reconstruction of the amino acids loss
        recon_aa_loss = self.recon_aa_loss(predictions=aa_recon, targets=aa_gt)

        #kl divergence loss
        # kl_loss = self.kl_divergence(mu, logvar)
        total_loss = self.beta_loss * recon_loss + self.gamma * recon_coords_loss +(1-self.beta_loss-self.gamma) * recon_aa_loss
        # total_loss = self.alpha * main_loss + self.beta_loss * recon_loss + self.gamma * recon_coords_loss +(1-self.alpha-self.beta_loss-self.gamma) * recon_aa_loss
        # total_loss = self.alpha * main_loss + self.beta_loss * recon_loss + self.gamma * recon_coords_loss + self.delta * recon_aa_loss + (1-self.alpha-self.beta_loss-self.gamma-self.delta) * kl_loss
        # total_loss = self.alpha*main_loss + recon_loss #+ recon_coords_loss
        # total_loss = main_loss
        # print("Total loss: ", total_loss)
        # print("Main loss: ", main_loss)
        # print("Recon loss: ", recon_loss)
        log_losses = {
            'total_loss': total_loss.detach(),
            # 'time_loss': main_loss.detach(),
            'scatter_recon_loss': recon_loss.detach(),
            'coords_recon_loss': recon_coords_loss.detach(),
            'aa_recon_loss': recon_aa_loss.detach(),
            # 'kl_loss': kl_loss.detach()
        }

        return total_loss, log_losses

    def main_loss(self, predictions, targets, valid_step=False):
        """
        takes in predictions and targets

        predictions and targets can be a list

        Args:
            predictions (dict): [description]
            targets (dict): [description]

        Returns:
            loss value
        """
        raise NotImplementedError

    def recon_loss(self, predictions, targets, valid_step=False):
        """
        takes in predictions and targets

        predictions and targets can be a list

        Args:
            predictions (dict): [description]
            targets (dict): [description]

        Returns:
            loss value
        """
        raise NotImplementedError

    def forward(self, batch):
        """
        the forward method should just take data
        as an argument


        should return:
            -  predictions
            -  embeddings
        """
        return NotImplementedError
    
class TGTransformerBaseModel_ATLAS_prop(LightningModule):
    """
    base model for models which take in
    torch geometric batches

    """

    def __init__(self, hparams=None):
        super(TGTransformerBaseModel_ATLAS_prop, self).__init__()

        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)

        self.save_hyperparameters()

        #self.hparams = hparams

        self.lr = hparams.lr
        self.task = hparams.task

    def configure_optimizers(self):
        # print("Going here")
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)

        lr_sched = CosineWarmupScheduler(opt, warmup=5000, max_iters=50000)

        return [opt], [lr_sched]

    def shared_step(self, batch):
        # print(batch)
        # import pdb; pdb.set_trace() 
        x, y = batch.x, batch.coords
        coords = batch.coords
        # import pdb; pdb.set_trace()
        x = x.float()#, y.float()

        preds, _, coeffs_recon, coeffs, _,_, coords_recon = self(batch)
        targets = y

        return preds, targets, coeffs_recon, coeffs, coords, coords_recon

    def relabel(self, loss_dict, label):

        loss_dict = {label + str(key): val for key, val in loss_dict.items()}

        return loss_dict

    def training_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        preds, targets, coeffs_recon, coeffs, coords, coords_recon = self.shared_step(batch)

        assert len(preds) == len(
            targets), f'preds: {len(preds)} targs: {len(targets)}'

        train_loss, train_loss_logs = self.multi_loss(
            predictions=preds, targets=targets, coeffs_recon=coeffs_recon, coeffs=coeffs, batch_idx=batch_idx,
              coords=coords, coords_recon=coords_recon)

        train_loss_logs = self.relabel(train_loss_logs, 'train_')

        self.log_dict(train_loss_logs, on_step=False, on_epoch=True, batch_size=self.batch_size)

        return train_loss

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        preds, targets, coeffs_recon, coeffs, coords, coords_recon = self.shared_step(batch)
        # print(targets)
        assert len(preds) == len(
            targets), f'preds: {len(preds)} targs: {len(targets)}'

        val_loss, val_loss_logs = self.multi_loss(
            predictions=preds, targets=targets, coeffs_recon=coeffs_recon, coeffs=coeffs, batch_idx=batch_idx,
              coords=coords, coords_recon=coords_recon)

        val_loss_logs = self.relabel(val_loss_logs, 'val_')

        self.log_dict(val_loss_logs, on_step=False, on_epoch=True, batch_size=self.batch_size)

        return val_loss

    def multi_loss(self, predictions, targets, coeffs_recon, coeffs, batch_idx, coords, coords_recon):
        # main loss
        main_loss = self.main_loss(predictions=predictions, targets=targets)

        # reconstruction loss
        recon_loss = self.recon_loss(predictions=coeffs_recon, targets=coeffs)

        #reconstruction of coordinates loss
        # recon_coords_loss = self.recon_coords_loss(predictions=coords_recon, targets=coords)

        # total_loss = self.alpha * main_loss + self.beta_loss * recon_loss + (1-self.alpha-self.beta_loss) * recon_coords_loss
        total_loss = self.alpha*main_loss + recon_loss #+ recon_coords_loss
        # total_loss = main_loss
        # print("Total loss: ", total_loss)
        # print("Main loss: ", main_loss)
        # print("Recon loss: ", recon_loss)
        log_losses = {
            'loss': total_loss.detach(),
            'main_loss': main_loss.detach(),
            'recon_loss': recon_loss.detach(),
            #'recon_coords_loss': recon_coords_loss.detach(),
        }

        return total_loss, log_losses

    def main_loss(self, predictions, targets, valid_step=False):
        """
        takes in predictions and targets

        predictions and targets can be a list

        Args:
            predictions (dict): [description]
            targets (dict): [description]

        Returns:
            loss value
        """
        raise NotImplementedError

    def recon_loss(self, predictions, targets, valid_step=False):
        """
        takes in predictions and targets

        predictions and targets can be a list

        Args:
            predictions (dict): [description]
            targets (dict): [description]

        Returns:
            loss value
        """
        raise NotImplementedError

    def forward(self, batch):
        """
        the forward method should just take data
        as an argument


        should return:
            -  predictions
            -  embeddings
        """
        return NotImplementedError

class TGTransformerBaseModel_ATLAS_noT(LightningModule):
    """
    base model for models which take in
    torch geometric batches

    """

    def __init__(self, hparams=None):
        super(TGTransformerBaseModel_ATLAS_noT, self).__init__()

        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)

        self.save_hyperparameters()

        #self.hparams = hparams

        self.lr = hparams.lr
        self.task = hparams.task

    def configure_optimizers(self):
        # print("Going here")
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)

        lr_sched = CosineWarmupScheduler(opt, warmup=5000, max_iters=50000)

        return [opt], [lr_sched]

    def shared_step(self, batch):
        # print(batch)
        # import pdb; pdb.set_trace() 
        x, y = batch.x, batch.time
        coords = batch.coords
        # import pdb; pdb.set_trace()
        x = x.float()#, y.float()

        preds, _, coeffs_recon, coeffs, _, coords_recon = self(batch)
        targets = y

        return preds, targets, coeffs_recon, coeffs, coords, coords_recon

    def relabel(self, loss_dict, label):

        loss_dict = {label + str(key): val for key, val in loss_dict.items()}

        return loss_dict

    def training_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        preds, targets, coeffs_recon, coeffs, coords, coords_recon = self.shared_step(batch)

        assert len(preds) == len(
            targets), f'preds: {len(preds)} targs: {len(targets)}'

        train_loss, train_loss_logs = self.multi_loss(
            predictions=preds, targets=targets, coeffs_recon=coeffs_recon, coeffs=coeffs, batch_idx=batch_idx,
              coords=coords, coords_recon=coords_recon)

        train_loss_logs = self.relabel(train_loss_logs, 'train_')

        self.log_dict(train_loss_logs, on_step=False, on_epoch=True, batch_size=self.batch_size)

        return train_loss

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        preds, targets, coeffs_recon, coeffs, coords, coords_recon = self.shared_step(batch)
        # print(targets)
        assert len(preds) == len(
            targets), f'preds: {len(preds)} targs: {len(targets)}'

        val_loss, val_loss_logs = self.multi_loss(
            predictions=preds, targets=targets, coeffs_recon=coeffs_recon, coeffs=coeffs, batch_idx=batch_idx,
              coords=coords, coords_recon=coords_recon)

        val_loss_logs = self.relabel(val_loss_logs, 'val_')

        self.log_dict(val_loss_logs, on_step=False, on_epoch=True, batch_size=self.batch_size)

        return val_loss

    def multi_loss(self, predictions, targets, coeffs_recon, coeffs, batch_idx, coords, coords_recon):
        # main loss
        main_loss = self.main_loss(predictions=predictions, targets=targets)

        # reconstruction loss
        recon_loss = self.recon_loss(predictions=coeffs_recon, targets=coeffs)

        #reconstruction of coordinates loss
        recon_coords_loss = self.recon_coords_loss(predictions=coords_recon, targets=coords)

        #reconstruction of the amino acids loss
        # recon_aa_loss = self.recon_aa_loss(predictions=aa_recon, targets=aa)

        total_loss = self.alpha * main_loss + self.beta_loss * recon_loss + (1-self.alpha-self.beta_loss) * recon_coords_loss
        # total_loss = self.alpha*main_loss + recon_loss #+ recon_coords_loss
        # total_loss = main_loss
        # print("Total loss: ", total_loss)
        # print("Main loss: ", main_loss)
        # print("Recon loss: ", recon_loss)
        log_losses = {
            'total_loss': total_loss.detach(),
            'time_loss': main_loss.detach(),
            'scatter_recon_loss': recon_loss.detach(),
            'coords_recon_loss': recon_coords_loss.detach(),
        }

        return total_loss, log_losses

    def main_loss(self, predictions, targets, valid_step=False):
        """
        takes in predictions and targets

        predictions and targets can be a list

        Args:
            predictions (dict): [description]
            targets (dict): [description]

        Returns:
            loss value
        """
        raise NotImplementedError

    def recon_loss(self, predictions, targets, valid_step=False):
        """
        takes in predictions and targets

        predictions and targets can be a list

        Args:
            predictions (dict): [description]
            targets (dict): [description]

        Returns:
            loss value
        """
        raise NotImplementedError

    def forward(self, batch):
        """
        the forward method should just take data
        as an argument


        should return:
            -  predictions
            -  embeddings
        """
        return NotImplementedError