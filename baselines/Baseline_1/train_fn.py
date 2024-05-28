"""
Function to train/fine-tune a PyTorch model, 
using an Accelerator wrapper for CPU-GPU(s)
support.

This training function only saves the best model
(i.e. when a main validation metric is surpassed by a 
margin, after the burn-in number of epochs).

Some notes on usage:
- Set training `args` in `TrainArgs.py` (BURNIN_N_EPOCHS, MAIN_METRIC, etc.)
- Must import `utilities.py` file with `pickle_obj`, `EpochCounter`.
- Set an optional `snapshot_path` to load an accelerate model snapshot and 
  resume training; else leave `None`.
- The loss function must be an attrib. of the model, to get placed on the 
  device properly with accelerate.
- The `dataloaders` passed must be a dictionary of Dataloaders with 'train' 
  and 'valid' keys.
"""

import utilities as U
import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator


def train_model(args,
                model,
                dataloaders,
                optimizer,
                stop_rule='no_improvement', # `None` for no stop rule
                snapshot_path=None,
                save_states=True,
                return_objs=False,
                use_acc_print=False,
                verbosity=0):

    """
    INITIALIZE DIRS, WEIGHTS, METRICS
    """
    if verbosity > 0:
        print('save_states:', save_states)
    best_model_wts = copy.deepcopy(model.state_dict())
    if (args.MODEL_SAVE_DIR is not None) and (args.MODEL_SAVE_DIR != ""):
        os.makedirs(args.MODEL_SAVE_DIR, exist_ok=True)
        
    # store metrics by epoch in list of dicts 
    # ('records' -> easy to convert to pd.DataFrame)
    records = []
    
    # initialize EpochCounter
    epoch_ctr = U.EpochCounter(0, args.MAIN_METRIC)


    """
    ACCELERATOR WRAPPER
    """
    acc = Accelerator(cpu=args.ON_CPU)
    (model, 
     optimizer, 
     dataloaders['train'],
     dataloaders['valid']) = acc.prepare(model, 
                                         optimizer, 
                                         dataloaders['train'], 
                                         dataloaders['valid'])
    # custom objects must be 'registered for checkpointing'
    acc.register_for_checkpointing(epoch_ctr)

    """
    INNER FUNCTIONS
    """
    def _save_snapshot(name):
        if (args.MODEL_SAVE_DIR is not None) and (args.MODEL_SAVE_DIR != ""):
            snapshot_path = f'{args.MODEL_SAVE_DIR}/{name}'
            acc.save_state(snapshot_path)

    def _log_output(out):
        if use_acc_print:
            acc.print(out)
        else:
            with open(args.PRINT_DIR, 'a') as f:
                f.write(out + "\n")
        
    """
    LOAD MODEL SNAPSHOT
    - optional, to resume training
    """
    if snapshot_path is not None:
        acc.load_state(snapshot_path)
        out = f"...resuming training from snapshot at epoch {epoch_ctr.n}"
        _log_output(out)
            
    """
    TRAINING LOOP
    """
    time_0 = time.time()
    ul_str = '-' * 10
    
    for epoch in range(epoch_ctr.n + 1, epoch_ctr.n + args.N_EPOCHS + 1):
        time_epoch_0 = time.time()
        epoch_ctr += 1
        out = f'\nEpoch {epoch}/{args.N_EPOCHS}\n{ul_str}'
        _log_output(out)
        if verbosity > 0:
            print(out)
        
        # each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if (phase == 'train'):
                model.train()
            else:
                model.eval()

            # loop through training/validation batches:
            for input_dict in dataloaders[phase]:
                optimizer.zero_grad()
                with torch.set_grad_enabled((phase == 'train')):
                    
                    # forward step
                    output_dict = model(input_dict) # calls model.forward
                    loss_dict = model.loss(input_dict, output_dict)

                    # train phase only: backward + optimizer steps
                    if (phase == 'train'):
                        acc.backward(loss_dict['loss'])
                        optimizer.step()

                    # update metrics
                    model.update_metrics(phase, 
                                         loss_dict, 
                                         input_dict, 
                                         output_dict)

        # after both train and valid phases are complete
        # calc losses/metrics
        epoch_hist_d = model.calc_metrics(epoch, input_dict)

        # log/print losses
        train_loss = epoch_hist_d['loss_train']
        valid_loss = epoch_hist_d['loss_valid']
        out = f'train loss: {train_loss:.6f} | valid loss: {valid_loss:.6f}'
        _log_output(out)
        if verbosity > 0:
            print(out)

        # check for new best validation loss
        after_burnin = epoch > args.BURNIN_N_EPOCHS
        
        if epoch_ctr.n == 1:
            epoch_ctr.set_best('_valid_loss', epoch, valid_loss)
            num_epochs_no_vl_improv = 0
        elif valid_loss < epoch_ctr.best['_valid_loss']['score']:
            epoch_ctr.set_best('_valid_loss', epoch, valid_loss)
            num_epochs_no_vl_improv = 0
        else:
            num_epochs_no_vl_improv += 1

        if after_burnin and stop_rule is not None:
            if stop_rule == 'no_improvement' \
            and num_epochs_no_vl_improv == args.NO_VALID_LOSS_IMPROVE_PATIENCE:
                out = f'Validation loss did not improve for {num_epochs_no_vl_improv} epochs: stopping.'
                print(out)
                _log_output(out)
                break
            
        # check for new best key validation score (by a margin)
        new_best_score, score_thresh_reached = False, False
        epoch_score = epoch_hist_d[args.MAIN_METRIC]
        # set initial score to beat, and validation loss, after first epoch
        if epoch_ctr.n == 1:
            epoch_ctr.set_best(args.MAIN_METRIC, epoch, epoch_score)
        best_main_metric = epoch_ctr.best[args.MAIN_METRIC]['score']
            
        score_thresh = best_main_metric * args.MAIN_METRIC_REL_IMPROV_THRESH
        if args.MAIN_METRIC_IS_BETTER == 'lower':
            score_thresh_reached = (epoch_score < score_thresh)
        elif args.MAIN_METRIC_IS_BETTER == 'higher':
            score_thresh_reached = (epoch_score > score_thresh)

        # if new best validation score threshold reached, record it
        if after_burnin and score_thresh_reached:
            new_best_score = True
            best_main_metric = epoch_hist_d[args.MAIN_METRIC]
            epoch_ctr.set_best(args.MAIN_METRIC, epoch, best_main_metric)
            epoch_key = f"epoch_{epoch}"

        # include epoch training time and reigning epoch with best validation score
        epoch_hist_d['sec_elapsed'] = round(time.time() - time_epoch_0, 2)
        epoch_hist_d['best_epoch'] = epoch_ctr.best[args.MAIN_METRIC]['epoch']

        # finally, append this epoch's losses, metrics, and time elapsed to records
        records.append(epoch_hist_d)

        # if new best epoch, log and save overwrite best saved model
        if new_best_score:
            # print msg
            score_str = f"{args.MAIN_METRIC}={epoch_hist_d[args.MAIN_METRIC]:.4f}"
            out = f"-> New best model! {score_str}"
            _log_output(out)
            if verbosity > 0:
                print(out)
            if save_states:
                print('Saving model...')
                # save 'best' model and training logs to reach it
                _save_snapshot('best')
                U.pickle_obj(args.TRAIN_LOGS_SAVE_DIR, records)
                if verbosity > 0:
                    _log_output('Model saved.')
            if return_objs:
                best_model_wts = copy.deepcopy(model.state_dict())

    """
    POST-TRAINING
    """
    # get total time elapsed
    t_min, t_sec = U.get_time_min_sec(time.time(), time_0)
    out = f'\n{epoch_ctr.n} epochs complete in {t_min:.0f}min, {t_sec:.1f}sec.\n'
    _log_output(out)
    print(out)

    # log final best validation score and epoch
    if epoch_ctr.n > args.BURNIN_N_EPOCHS:
        best_epoch = epoch_ctr.best[args.MAIN_METRIC]['epoch']
        out = f'Best {args.MAIN_METRIC}: {best_main_metric:.4f} at epoch {best_epoch}\n'
        _log_output(out)

    # save final training log
    U.pickle_obj(args.TRAIN_LOGS_SAVE_DIR, records)

    # optional: save final epoch's model state
    if save_states and args.SAVE_FINAL_MODEL:
        _save_snapshot('final')

    # optional: load best model weights and return tuple with history log
    if return_objs:
        if best_model_wts is not None:
            model.load_state_dict(best_model_wts)
            return (model, records)
        else:
            out = 'No best model found; no weights saved!'
            _log_output(out)
            print(out)
            return (None, None)
    else:
        return (None, None)

