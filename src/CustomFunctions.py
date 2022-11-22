import json
# import string
from functools import partial
from typing import List, Dict, Union, Optional
from joblib import dump, load

import numpy as np
import torch
import time
import torch.nn as nn
from pandas import DataFrame
from ray.tune import CLIReporter
from ray.tune.logger import UnifiedLogger
# import networkx as nx
from ray.tune.trial import Trial
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.sparse import csr_matrix
import pandas


class WriterCLIReporter(CLIReporter):
    def __init__(self, checkpoint_dir: str,
                 metric_columns: Union[None, List[str], Dict[str, str]] = None,
                 parameter_columns: Union[None, List[str], Dict[str, str]] = None,
                 total_samples: Optional[int] = None,
                 max_progress_rows: int = 20,
                 max_error_rows: int = 20,
                 max_report_frequency: int = 5,
                 infer_limit: int = 3,
                 print_intermediate_tables: Optional[bool] = None,
                 metric: Optional[str] = None,
                 mode: Optional[str] = None):
        self.checkpoint_dir = checkpoint_dir
        # self.best_config_file_name = best_config_file_name
        super(CLIReporter, self).__init__(
            metric_columns, parameter_columns, total_samples,
            max_progress_rows, max_error_rows, max_report_frequency,
            infer_limit, print_intermediate_tables, metric, mode)
        super().__init__()

    def write_best_config(self, trials):
        current_best_trial, metric = self._current_best_trial(trials)

        if current_best_trial is None:
            return
        print("Writing best config to file...")
        # checkpoint_dir = current_best_trial.custom_dirname
        config = current_best_trial.last_result.get("config", {})
        # parameter_columns = parameter_columns or list(config.keys())
        # if isinstance(parameter_columns, Mapping):
        #     parameter_columns = parameter_columns.keys()
        # params = {p: config.get(p) for p in parameter_columns}

        with open("/scratch/l/lstein/ftaj/" + self.checkpoint_dir + '/best_config.json', 'w') as fp:
            json.dump(config, fp)

    def report(self, trials: List[Trial], done: bool, *sys_info: Dict):
        print(self._progress_str(trials, done, *sys_info))
        self.write_best_config(trials)


def model_save(cv_index, cur_epoch, cur_model, cur_optimizer, train_losses, valid_losses,
               all_avg_train_losses, all_avg_valid_losses, save_model_path, save_model_frequency,
               early_stopper, all_final_epochs,
               force: bool = False, sklearn: bool = False):
    if save_model_path is None:
        exit("Model Save Folder not provided, skipping model save...")
    elif sklearn is True and ((cur_epoch % save_model_frequency == 0) or force is True):
        dump(cur_model, save_model_path + "/checkpoint_cv_" + str(cv_index) + ".joblib")

    # Forcing save ignores checkpointing frequency
    elif (cur_epoch % save_model_frequency == 0) or force is True:
        print("Saving model at CV", cv_index, ", epoch", cur_epoch)
        torch.save({
            'cv': cv_index,
            'epoch': cur_epoch + 1,  # +1 to not redo the same epoch on resume
            'model_state_dict': cur_model.state_dict(),
            'optimizer_state_dict': cur_optimizer.state_dict(),
            'train_losses': train_losses,
            'valid_losses': valid_losses,
            'all_avg_train_losses': all_avg_train_losses,
            'all_avg_valid_losses': all_avg_valid_losses,
            'early_stopper': early_stopper,
            'all_final_epochs': all_final_epochs
        }, save_model_path + "/checkpoint_cv_" + str(cv_index) + ".pt")


def cell_drug_match(class_data: DataFrame):
    """
    This function finds the maximum number of semi-exclusive pairings between cell lines and drugs, where each drugs
    can have a maximum of n cell lines assigned to it, but each cell line is only assigned to a single drug.

    :param class_data:
    :return:
    """
    class_net_subset = class_data[['ccl_name', 'cpd_name']]

    df = pandas.crosstab(class_net_subset.ccl_name, class_net_subset.cpd_name)
    cur_drugs = df.columns.tolist()

    # Duplicate drug columns (n times) to allow for mutliple cell-line/drug assignments
    # TODO generalize this scheme to n duplications
    for col_name in cur_drugs:
        df['duplicate_' + col_name] = df[col_name]

    cur_cells = df.index.tolist()

    # Each cell is matched to a drug
    csr = csr_matrix(df)
    max_match = maximum_bipartite_matching(csr, perm_type='column')
    max_match_list = max_match.tolist()

    # Create dictionary of matches based on drugs
    match_dict = {}
    for i in range(len(max_match_list)):
        # if 'duplicate' in df.columns[max_match_list[i]]:
        #     break
        cur_drug_name = df.columns[max_match_list[i]].replace('duplicate_', '')
        if cur_drug_name in match_dict.keys():
            match_dict[cur_drug_name].append(cur_cells[i])
        else:
            match_dict[cur_drug_name] = [cur_cells[i]]

    return match_dict


def produce_amount_keys(amount_of_keys, length=512):
    keys = set()
    pickchar = partial(
        np.random.choice,
        np.array(['0', '1']))
    while len(keys) < amount_of_keys:
        keys |= {''.join([pickchar() for _ in range(length)]) for _ in range(amount_of_keys - len(keys))}
    return keys


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, train_idx=None, valid_idx=None, patience=7, save=False, save_after_n_epochs=5, lower_better=True,
                 verbose=False, delta=0.001, path='checkpoint.pt', trace_func=print, best_score=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 3
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = best_score
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.save = save
        self.lower_better = lower_better
        self.save_after_n_epochs = save_after_n_epochs
        self.best_epoch = 0

    def __call__(self, score, epoch, model=None, optimizer=None):

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if self.save is True:
                # Don't save in the first N epochs
                if epoch > self.save_after_n_epochs:
                    self.save_checkpoint(score, model, optimizer, epoch)
            return

        if self.lower_better is True:
            # score should be lower than best minus delta
            if score > (self.best_score - self.delta):
                self.counter += 1
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.best_epoch = epoch
                if self.save is True:
                    # Don't save in the first N epochs
                    if epoch > self.save_after_n_epochs:
                        self.save_checkpoint(score, model, optimizer, epoch)
                self.counter = 0

        else:
            # Higher score is better, score should be higher than best plus delta
            if score < (self.best_score + self.delta):
                self.counter += 1
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.best_epoch = epoch
                if self.save is True:
                    # Don't save in the first N epochs
                    if epoch > self.save_after_n_epochs:
                        self.save_checkpoint(score, model, optimizer, epoch)
                self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, epoch):
        """Saves model when validation loss decreases."""
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        start_time = time.time()
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'train_idx': self.train_idx,
            'valid_idx': self.valid_idx
            # 'amp': amp.state_dict(),
        }, self.path)
        print("Saving done in", time.time() - start_time, "seconds")
        self.val_loss_min = val_loss


class AutoEncoderEarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=3, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, best_score=None,
                 ignore=5):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 3
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = best_score
        self.ignore = ignore
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, train_loss, model, optimizer, epoch):

        score = -train_loss

        if self.best_score is None:
            self.best_score = score
            # Don't save in the first N epochs
            if epoch > self.ignore:
                self.save_checkpoint(train_loss, model, optimizer, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if epoch > self.ignore:
                self.save_checkpoint(train_loss, model, optimizer, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, epoch):
        """Saves model when validation loss decreases."""
        if self.verbose:
            self.trace_func(f'Training loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        start_time = time.time()
        torch.save({
            'epoch': epoch + 1,
            'cur_model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': val_loss,
            # 'amp': amp.state_dict(),
        }, self.path)
        print("Saving done in", time.time() - start_time, "seconds")
        self.val_loss_min = val_loss


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or \
            isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
        m.reset_parameters()


def init_module_weights(module, init_func, bias=0.01):
    if isinstance(module, nn.Linear):
        init_func(module.weight)
        module.bias.data.fill_(bias)


class MyLoggerCreator(UnifiedLogger):

    def __init__(self, config, logdir: str = "/scratch/l/lstein/ftaj/ray_logdir"):
        super().__init__(config, logdir)
