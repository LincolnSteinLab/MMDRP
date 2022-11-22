import copy
# import re
import sys
import time
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch_geometric
from captum.attr import IntegratedGradients
from joblib import load
from sklearn.metrics import mean_squared_error
from torch import optim
from torch.utils import data
from torch.utils.data import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
# from torch_geometric.data import DataLoader
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

from CustomFunctions import AverageMeter, EarlyStopping, model_save
from DataImportModules import AutoEncoderPrefetcher, DataPrefetcher, GNNDataPrefetcher, MyGNNData, GenFeatures
from Models import LMFTest
from drug_visualization import drug_interpret_viz


def morgan_train(cur_model, criterion, optimizer, epoch, train_loader=None, valid_loader=None, verbose: bool = False):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    train_losses = AverageMeter('Training Loss', ':.4e')
    valid_losses = AverageMeter('Validation Loss', ':.4e')

    prefetcher = AutoEncoderPrefetcher(train_loader)
    cur_data = next(prefetcher)
    # Training
    cur_model.train()
    end = time.time()
    data_end_time = time.time()

    if verbose:
        print("Training Drug Model...")
    i = 0
    running_loss = 0.0
    while cur_data is not None:
        i += 1
        data_time.update(time.time() - data_end_time)

        # forward + backward + optimize
        output = cur_model(cur_data)
        loss = criterion(output, cur_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.update(loss.item(), cur_data.shape[0])

        data_end_time = time.time()
        cur_data = next(prefetcher)

        # Calculating running loss results in a CPU transfer + GPU sync, which slows down training
        running_loss += loss.item()
        if verbose:
            if i % 1000 == 999:  # print every 1000 mini-batches
                print('[%d, %5d] train loss: %.3f' %
                      (epoch, i, running_loss / 1000))
                running_loss = 0.0

    if valid_loader is None:
        return train_losses
    else:
        prefetcher = AutoEncoderPrefetcher(valid_loader)
        cur_data = next(prefetcher)
        # Training
        cur_model.eval()
        end = time.time()
        data_end_time = time.time()

        if verbose:
            print("Validating Drug Model...")
        i = 0
        running_loss = 0.0
        with torch.no_grad():
            while cur_data is not None:
                i += 1
                data_time.update(time.time() - data_end_time)

                # forward + backward + optimize
                output = cur_model(cur_data)
                valid_loss = criterion(output, cur_data)

                valid_losses.update(valid_loss.item(), cur_data.shape[0])
                data_end_time = time.time()

                # Calculating running loss results in a CPU transfer + GPU sync, which slows down training
                running_loss += valid_loss.item()
                if verbose:
                    if i % 200 == 199:  # print every 200 mini-batches
                        print('[%d, %5d] valid loss: %.3f' %
                              (epoch, i, running_loss / 200))
                        running_loss = 0.0

                data_end_time = time.time()
                cur_data = next(prefetcher)

    if verbose:
        print("Sum train and valid losses at epoch", epoch, ":", train_losses.sum, valid_losses.sum)

    return train_losses, valid_losses


def omic_train(cur_model, criterion, optimizer, epoch, train_loader=None, valid_loader=None, to_gpu: bool = True,
               verbose: bool = False, retain_grad=False):
    batch_time = AverageMeter('Time', ':6.3f')
    # data_time = AverageMeter('Data', ':6.3f')
    train_losses = AverageMeter('Training Loss', ':.4e')
    valid_losses = AverageMeter('Validation Loss', ':.4e')

    if train_loader is not None:
        # === Train Model ===
        if to_gpu is True:
            # Use data prefetching
            prefetcher = AutoEncoderPrefetcher(train_loader)
        else:
            prefetcher = iter(train_loader)

        cur_data = next(prefetcher)

        # Training
        cur_model.train()
        end = time.time()
        # data_end_time = time.time()

        if verbose:
            print("Training Omic Model...")
        i = 0
        # running_loss = 0.0
        while cur_data is not None:
            i += 1
            # data_time.update(time.time() - data_end_time)

            # forward + backward + optimize
            output = cur_model(cur_data)
            loss = criterion(output, cur_data)
            train_losses.update(loss.item(), cur_data.shape[0])

            try:
                cur_data = next(prefetcher)
            except:
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # data_end_time = time.time()

            # Calculating running loss results in a CPU transfer + GPU sync, which slows down training
            # running_loss += loss.item()
            # if verbose:
            #     if i % 1000 == 999:  # print every 1000 mini-batches
            #         print('[%d, %5d] train loss: %.3f' %
            #               (epoch, i, running_loss / 1000))
            #         running_loss = 0.0

    if valid_loader is None:
        return train_losses
    else:
        if to_gpu is True:
            prefetcher = AutoEncoderPrefetcher(valid_loader)
        else:
            prefetcher = iter(valid_loader)

        cur_data = next(prefetcher)

        # Training
        cur_model.eval()
        end = time.time()
        # data_end_time = time.time()

        if verbose:
            print("Validating Omic Model...")
        i = 0
        # running_loss = 0.0
        with torch.no_grad():
            while cur_data is not None:
                i += 1
                # data_time.update(time.time() - data_end_time)

                # forward + backward + optimize
                output = cur_model(cur_data)
                valid_loss = criterion(output, cur_data)

                valid_losses.update(valid_loss.item(), cur_data.shape[0])
                # data_end_time = time.time()
                try:
                    cur_data = next(prefetcher)
                except:
                    break

                # running_loss += valid_loss.item()

                # if i % 200 == 199:  # print every 200 mini-batches
                #     print('[%d, %5d] valid loss: %.3f' %
                #           (epoch, i, running_loss / 200))
                #     running_loss = 0.0

    if verbose:
        print("Sum train and valid losses at epoch", epoch, ":", train_losses.sum, valid_losses.sum)

    return train_losses, valid_losses


def elasticnet_drp_train(cur_model, criterion, optimizer, epoch, train_loader=None, valid_loader=None,
                         return_results: bool = False,
                         verbose: bool = False):
    data_time = AverageMeter('Data', ':6.3f')
    train_losses = AverageMeter('Training Loss', ':.4e')
    valid_losses = AverageMeter('Validation Loss', ':.4e')
    if train_loader is None and valid_loader is None:
        raise AssertionError("At least one of train loader or valid loader must be provided!")

    if train_loader is not None:
        len_train_loader = len(train_loader)
        # ==== Train Model ====
        # switch to train mode
        end = time.time()
        data_end_time = time.time()

        # prefetcher = DataPrefetcher(train_loader)
        prefetcher = iter(train_loader)
        _, cur_dr_data, cur_dr_target, cur_loss_weights = next(prefetcher)
        # Check data format once before training

        if verbose:
            print("Training DRP Model...")
        # with profiler.profile(record_shapes=False, use_cuda=True, profile_memory=True) as prof:
        #   with profiler.record_function("model_inference"):
        i = 0
        running_loss = 0.0
        start_time = time.time()
        try:
            while cur_dr_data is not None:
                i += 1

                # print("Data generation time for batch", i, ":", time.time() - data_end_time)
                # data_time.update(time.time() - data_end_time)

                # Assume that we only have gene expression and Morgan data; must concatenate
                cur_dr_data = torch.cat((cur_dr_data[0], cur_dr_data[1]), dim=1)
                cur_dr_data = cur_dr_data.numpy()
                cur_dr_target = cur_dr_target.numpy().ravel()

                # print("cur_dr_data shape:", cur_dr_data.shape)
                # print("cur_dr_target shape:", cur_dr_target.shape)
                # print(cur_dr_target)
                # exit(0)
                cur_model.partial_fit(cur_dr_data, cur_dr_target)
                # output = cur_model(*cur_dr_data)
                # loss = criterion(output, cur_dr_target, cur_loss_weights)

                if i % 1000 == 999:
                    #     # Sample losses every 1000 batches
                    #     train_losses.update(loss.item(), cur_dr_target.shape[0])
                    #     running_loss = loss.item()
                    if verbose:
                        # print every 1000 mini-batches
                        print('[%d, %5d / %5d, %5d s] 1000 batches done!' %
                              (epoch, i, len_train_loader, time.time() - start_time))
                    #         # print('[%d, %5d / %5d, %5d s]' %
                    #         #       (epoch, i, len_train_loader, time.time() - start_time))
                    #     # running_loss = 0.0
                    start_time = time.time()

                # data_end_time = time.time()
                _, cur_dr_data, cur_dr_target, cur_loss_weights = next(prefetcher)

        except StopIteration:
            pass

    if valid_loader is None:
        return train_losses

    else:
        len_valid_loader = len(valid_loader)
        # ==== Validate Model ====
        # switch to evaluation mode
        # prefetcher = DataPrefetcher(valid_loader)
        prefetcher = iter(valid_loader)
        cur_sample_info, cur_dr_data, cur_dr_target, cur_loss_weights = next(prefetcher)

        if verbose:
            print("Validating DRP Model...")
        # with profiler.profile(record_shapes=False, use_cuda=True, profile_memory=True) as prof:
        #   with profiler.record_function("model_inference"):
        all_results = []
        i = 0
        running_loss = 0.0
        # with torch.no_grad():
        # batch_start = time.time()
        # end = time.time()
        try:
            while cur_dr_data is not None:
                i += 1

                cur_dr_data = torch.cat((cur_dr_data[0], cur_dr_data[1]), dim=1)
                cur_dr_data = cur_dr_data.numpy()
                cur_dr_target = cur_dr_target.numpy().ravel()

                cur_predictions = cur_model.predict(cur_dr_data)
                valid_loss = mean_squared_error(y_true=cur_dr_target, y_pred=cur_predictions,
                                                multioutput='raw_values', squared=False)
                # forward + loss measurement
                # output = cur_model(*cur_dr_data)
                # valid_loss = criterion(output, cur_dr_target, cur_loss_weights)
                # Second argument is effectively the batch size
                valid_losses.update(float(valid_loss), cur_dr_target.shape[0])

                if return_results is True:
                    cur_targets = cur_dr_target.tolist()
                    cur_preds = cur_predictions.tolist()

                    valid_loss = np.absolute(cur_dr_target - cur_predictions)
                    cur_losses = valid_loss.tolist()

                    # print("cur_targets:", cur_targets)
                    # print("cur_preds:", cur_preds)
                    # print("cur_losses:", cur_losses)
                    # exit(0)
                    # cur_targets = [target[0] for target in cur_targets]
                    # cur_preds = [pred[0] for pred in cur_preds]
                    # cur_losses = [loss[0] for loss in cur_losses]
                    cur_dict = {
                        'cpd_name': cur_sample_info['drug_name'],
                        'cell_name': cur_sample_info['cell_line_name'],
                        'target': cur_targets,
                        'predicted': cur_preds,
                        'rmse_loss': cur_losses
                    }
                    all_results.append(pd.DataFrame.from_dict(cur_dict))

                cur_sample_info, cur_dr_data, cur_dr_target, cur_loss_weights = next(prefetcher)
                running_loss += np.mean(valid_loss)
                if verbose:
                    if i % 200 == 199:  # print every 200 mini-batches
                        print('[%d, %5d / %5d] running valid loss: %.6f' %
                              (epoch, i, len_valid_loader, running_loss / 200))
                        running_loss = 0.0
        except StopIteration:
            pass

        # if verbose:
        #     print("Sum train and valid losses at epoch", epoch, ":", train_losses.sum, valid_losses.sum)

        if return_results:
            return train_losses, valid_losses, pd.concat(all_results)
        else:
            return train_losses, valid_losses


def drp_train(cur_model, criterion, optimizer, epoch, train_loader=None, valid_loader=None,
              return_results: bool = False,
              verbose: bool = False):
    data_time = AverageMeter('Data', ':6.3f')
    train_losses = AverageMeter('Training Loss', ':.4e')
    valid_losses = AverageMeter('Validation Loss', ':.4e')
    if train_loader is None and valid_loader is None:
        raise AssertionError("At least one of train loader or valid loader must be provided!")

    if train_loader is not None:
        len_train_loader = len(train_loader)
        # ==== Train Model ====
        # switch to train mode
        cur_model.train()
        end = time.time()
        data_end_time = time.time()

        # prefetcher = DataPrefetcher(train_loader)
        prefetcher = iter(train_loader)
        _, cur_dr_data, cur_dr_target, cur_loss_weights = next(prefetcher)
        # Check data format once before training
        assert len(cur_dr_data) == cur_model.len_encoder_list, "DataPrefetcher DR data doesn't match # encoders"

        if verbose:
            print("Training DRP Model...")
        # with profiler.profile(record_shapes=False, use_cuda=True, profile_memory=True) as prof:
        #   with profiler.record_function("model_inference"):
        i = 0
        running_loss = 0.0
        start_time = time.time()
        try:
            while cur_dr_data is not None:
                i += 1

                # print("Data generation time for batch", i, ":", time.time() - data_end_time)
                data_time.update(time.time() - data_end_time)

                # forward + backward + optimize
                output = cur_model(*cur_dr_data)
                loss = criterion(output, cur_dr_target, cur_loss_weights)

                # By default, gradients are accumulated in buffers( i.e, not overwritten) whenever .backward()
                # is called, so must reset to zero manually if needed.
                optimizer.zero_grad()

                # Backward pass: compute gradient of the loss with respect to all the learnable
                # parameters of the model.The following is for mixed-precision training
                # with amp.scale_loss(loss, optimizer) as scaled_loss:
                #     scaled_loss.backward()
                loss.backward()
                # Calling the step function on an Optimizer makes an update to its
                # parameters
                optimizer.step()

                if i % 1000 == 999:
                    # Sample losses every 1000 batches
                    train_losses.update(loss.item(), cur_dr_target.shape[0])
                    running_loss = loss.item()
                    if verbose:
                        # print every 1000 mini-batches
                        print('[%d, %5d / %5d, %5d s] running train loss: %.6f' %
                              (epoch, i, len_train_loader, time.time() - start_time, running_loss))
                        # print('[%d, %5d / %5d, %5d s]' %
                        #       (epoch, i, len_train_loader, time.time() - start_time))
                    # running_loss = 0.0
                    start_time = time.time()

                data_end_time = time.time()
                _, cur_dr_data, cur_dr_target, cur_loss_weights = next(prefetcher)

        except StopIteration:
            pass

    if valid_loader is None:
        return train_losses

    else:
        len_valid_loader = len(valid_loader)
        # ==== Validate Model ====
        # switch to evaluation mode
        cur_model.eval()
        # prefetcher = DataPrefetcher(valid_loader)
        prefetcher = iter(valid_loader)
        cur_sample_info, cur_dr_data, cur_dr_target, cur_loss_weights = next(prefetcher)
        assert len(cur_dr_data) == cur_model.len_encoder_list, "DataPrefetcher DR data doesn't match # encoders"

        if verbose:
            print("Validating DRP Model...")
        # with profiler.profile(record_shapes=False, use_cuda=True, profile_memory=True) as prof:
        #   with profiler.record_function("model_inference"):
        all_results = []
        i = 0
        running_loss = 0.0
        with torch.no_grad():
            # batch_start = time.time()
            # end = time.time()
            try:
                while cur_dr_data is not None:
                    i += 1

                    # forward + loss measurement
                    output = cur_model(*cur_dr_data)
                    valid_loss = criterion(output, cur_dr_target, cur_loss_weights)
                    # Second argument is effectively the batch size
                    valid_losses.update(valid_loss.item(), cur_dr_target.shape[0])

                    if return_results is True:
                        cur_targets = cur_dr_target.tolist()
                        cur_targets = [target[0] for target in cur_targets]

                        cur_preds = output.tolist()
                        cur_preds = [pred[0] for pred in cur_preds]

                        cur_dict = {
                            'cpd_name': cur_sample_info['drug_name'],
                            'cell_name': cur_sample_info['cell_line_name'],
                            'target': cur_targets,
                            'predicted': cur_preds
                        }
                        all_results.append(pd.DataFrame.from_dict(cur_dict))

                    cur_sample_info, cur_dr_data, cur_dr_target, cur_loss_weights = next(prefetcher)
                    running_loss += valid_loss.item()
                    if verbose:
                        if i % 200 == 199:  # print every 200 mini-batches
                            print('[%d, %5d / %5d] running valid loss: %.6f' %
                                  (epoch, i, len_valid_loader, running_loss / 200))
                            running_loss = 0.0
            except StopIteration:
                pass

        # if verbose:
        #     print("Sum train and valid losses at epoch", epoch, ":", train_losses.sum, valid_losses.sum)

        if return_results:
            return train_losses, valid_losses, pd.concat(all_results)
        else:
            return train_losses, valid_losses


def gnn_drp_train(cur_model, criterion, optimizer, epoch, train_loader=None, valid_loader=None,
                  verbose: bool = False, return_results: bool = False, retain_grad: bool = False):
    data_time = AverageMeter('Data', ':6.3f')
    train_losses = AverageMeter('Training Loss', ':.4e')
    valid_losses = AverageMeter('Validation Loss', ':.4e')
    if train_loader is None and valid_loader is None:
        raise AssertionError("At least one of train loader or valid loader must be provided!")

    if train_loader is not None:
        len_train_loader = len(train_loader)
        # ==== Train Model ====
        # switch to train mode
        cur_model.train()
        end = time.time()
        data_end_time = time.time()

        train_loader_iter = iter(train_loader)
        cur_samples = next(train_loader_iter)
        # Check data format once before training
        assert len(cur_samples[2]) + 1 == cur_model.len_encoder_list, \
            "Training loader data doesn't match # encoders! Len Omic data:" + str(len(cur_samples[2])) + \
            ", Len encoder list:" + str(cur_model.len_encoder_list)

        if verbose:
            print("Training DRP Model...")
        # with profiler.profile(record_shapes=False, use_cuda=True, profile_memory=True) as prof:
        #   with profiler.record_function("model_inference"):
        i = 0
        running_loss = 0.0
        start_time = time.time()
        try:
            while True:
                i += 1

                # print("Data generation time for batch", i, ":", time.time() - data_end_time)
                data_time.update(time.time() - data_end_time)

                # forward + backward + optimize
                # Note: Do NOT use star expansion when passing a torch_geometric.data object
                output = cur_model(cur_samples[1], cur_samples[2])

                # TODO reshape until dataloader is fixed
                # cur_target = cur_data.target.reshape(cur_data.target.shape[0], 1)
                # cur_loss_weights = cur_data.loss_weight.reshape(cur_data.loss_weight.shape[0], 1)

                loss = criterion(output, cur_samples[3], cur_samples[4])

                cur_train_loss_item = loss.item()
                train_losses.update(cur_train_loss_item, cur_samples[3].shape[0])
                running_loss += cur_train_loss_item

                # By default, gradients are accumulated in buffers( i.e, not overwritten) whenever .backward()
                # is called, so must reset to zero manually if needed.
                optimizer.zero_grad()

                if retain_grad is True:
                    for name, param in cur_model.named_parameters():
                        # Must force the retention of gradients for intermediate layers
                        param.retain_grad()

                # Backward pass: compute gradient of the loss with respect to all the learnable
                # parameters of the model.The following is for mixed-precision training
                # with amp.scale_loss(loss, optimizer) as scaled_loss:
                #     scaled_loss.backward()
                loss.backward()

                # for name, param in cur_model.named_parameters():
                #     print(name, param.grad.abs().sum())
                # exit()

                # Calling the step function on an Optimizer makes an update to its
                # parameters
                optimizer.step()

                if i % 1000 == 999:
                    # Sample losses every 1000 batches
                    # train_losses.update(loss.item(), cur_data.shape[0])
                    running_loss = running_loss / 1000
                    if verbose:
                        # print every 1000 mini-batches
                        print('[%d, %5d / %5d, %5d s] running train loss: %.6f' %
                              (epoch, i, len_train_loader, time.time() - start_time, running_loss))
                        # print('[%d, %5d / %5d, %5d s]' %
                        #       (epoch, i, len_train_loader, time.time() - start_time))
                    running_loss = 0.0
                    start_time = time.time()

                # if i % 1000 == 0:  # print every 1000 mini-batches
                #     # train_r2.update(0, cur_dr_target.shape[0])
                #     # Note: if we use loss instead of loss.item(), the whole graph is retained, taking lots of RAM
                #     # Still, calling .item() results in a GPU to CPU transfer, causing a potential slowdown
                #
                #     # Measure elapsed time, must first sync cuda operations
                #     torch.cuda.synchronize()
                #     batch_time.update((time.time() - end))
                #     # train_progress.display(i)
                #     end = time.time()
                #     if args.benchmark is not None:
                #         if i == 1000:
                #             break

                data_end_time = time.time()
                cur_samples = next(train_loader_iter)

        except StopIteration:
            pass

    if valid_loader is None:
        return train_losses

    else:
        len_valid_loader = len(valid_loader)
        # ==== Validate Model ====
        # switch to evaluation mode
        cur_model.eval()
        valid_loader_iter = iter(valid_loader)
        cur_samples = next(valid_loader_iter)
        assert len(cur_samples[2]) + 1 == cur_model.len_encoder_list, \
            "Validation loader data doesn't match # encoders. Len Omic data:" + str(len(cur_samples[2])) + \
            ", Len encoder list:" + str(cur_model.len_encoder_list)

        if verbose:
            print("Validating DRP Model...")
        # with profiler.profile(record_shapes=False, use_cuda=True, profile_memory=True) as prof:
        #   with profiler.record_function("model_inference"):
        i = 0
        running_loss = 0.0
        all_results = []
        with torch.no_grad():
            # batch_start = time.time()
            # end = time.time()
            try:
                while True:
                    i += 1

                    # forward + loss measurement
                    output = cur_model(cur_samples[1], cur_samples[2])
                    # TODO reshape until dataloader is fixed
                    # cur_target = cur_data.target.reshape(cur_data.target.shape[0], 1)
                    # cur_loss_weights = cur_data.loss_weight.reshape(cur_data.loss_weight.shape[0], 1)
                    valid_loss = criterion(output, cur_samples[3], cur_samples[4])

                    # Second argument is effectively the batch size
                    cur_loss_item = valid_loss.item()
                    valid_losses.update(cur_loss_item, cur_samples[3].shape[0])

                    if return_results is True:
                        cur_targets = cur_samples[3].tolist()
                        cur_targets = [target[0] for target in cur_targets]

                        cur_preds = output.tolist()
                        cur_preds = [pred[0] for pred in cur_preds]

                        cur_dict = {
                            'cpd_name': cur_samples[0]['drug_name'],
                            'cell_name': cur_samples[0]['cell_line_name'],
                            'target': cur_targets,
                            'predicted': cur_preds
                        }
                        all_results.append(pd.DataFrame.from_dict(cur_dict))

                    cur_samples = next(valid_loader_iter)
                    running_loss += cur_loss_item

                    if i % 200 == 199:  # print every 200 mini-batches
                        if verbose:
                            print('[%d, %5d / %5d] valid loss: %.6f' %
                                  (epoch, i, len_valid_loader, running_loss / 200))
                        running_loss = 0.0

            except StopIteration:
                pass
        # if verbose:
        #     print("Sum train and valid losses at epoch", epoch, ":", train_losses.sum, valid_losses.sum)
        if return_results is True:
            return train_losses, valid_losses, pd.concat(all_results)
        else:
            return train_losses, valid_losses


def cross_validate(train_data, train_function, cv_folds, batch_size: int, cur_model, criterion,
                   patience: int = 5, delta: float = 0.0001, max_epochs: int = 100, learning_rate: float = 0.001,
                   final_full_train: bool = False, NUM_WORKERS: int = 0, theoretical_loss: bool = False,
                   train_only: bool = False, omic_standardize: bool = False, summary_writer: SummaryWriter = None,
                   save_cv_preds: bool = True, redo_validation: bool = False,
                   redo_interpretation: bool = None, min_target: float = None, dr_sub_cpd_names: [str] = None,
                   save_epoch_results: bool = False,
                   epoch_save_folder: str = None,
                   save_model: bool = False, save_model_frequency: int = None, save_model_path: str = None,
                   resume: bool = False, to_gpu: bool = True, verbose: bool = False):
    """
    This function will train the given model on len(cv_folds)-1 folds for max_epochs and then validates
    on the remaining fold.
    :cv_folds: a list of lists of format [[train_indices, valid_indices], ...] format
    :return: sum and average of train and validation losses for the current k-fold setup
    """
    max_final_epoch = max_epochs  # this will be changed if doing cross-validation
    avg_final_epoch = max_epochs  # ditto
    avg_train_losses, avg_valid_losses, avg_untrained_loss = 0.0, 0.0, 0.0
    if train_function == gnn_drp_train:
        # GNN data loaders handle graph data differently
        cur_data_loader = torch_geometric.data.DataLoader
    else:
        cur_data_loader = data.DataLoader

    n_folds = len(cv_folds)
    all_avg_train_losses, all_avg_valid_losses = [], []
    all_final_epochs = []
    all_untrained_losses = []
    cv_resume = resume
    if verbose:
        print("Training + Validation Data Length:", str(len(train_data)))

    # Transfer the model to CPU to save space on GPU
    if train_function != elasticnet_drp_train:
        cur_model.cpu()

    if not train_only:
        cv_index = 0
        cur_epoch = 1
        if cv_resume is True:
            print("Trying to load checkpoint...")
            if save_model_path is None:
                exit("Model path must be given for resume!")
            # Check folder for highest CV checkpoint
            cv_checkpoints = glob.glob(save_model_path + '/checkpoint_cv_*')
            if len(cv_checkpoints) == 0:
                print("Checkpoint not found! Starting from scratch...")
                cv_index = 0
                cur_epoch = 1
                cv_resume = False  # to not try to load model and optimizer state dicts

            else:
                all_cv_idx = [int(s) for checkpoint in cv_checkpoints for s in re.findall(r'\d+', checkpoint)]
                cv_index = max(all_cv_idx)
                print("Loading checkpoing from:", save_model_path + "/checkpoint_cv_" + str(cv_index) + ".pt")
                if train_function != elasticnet_drp_train:
                    # Load checkpoint and check for max epoch
                    cur_checkpoint = torch.load(save_model_path + "/checkpoint_cv_" + str(cv_index) + ".pt")
                    cur_epoch = cur_checkpoint['epoch']
                    all_avg_train_losses = cur_checkpoint['all_avg_train_losses']
                    all_avg_valid_losses = cur_checkpoint['all_avg_valid_losses']
                    print("Loaded checkpoint with CV:", cv_index, "and epoch:", cur_epoch)
                    print("\t-> Best validation loss:", cur_checkpoint['early_stopper'].best_score)
                    checkpoint_early_stopper = cur_checkpoint['early_stopper']
                    all_final_epochs = cur_checkpoint['all_final_epochs']

        while cv_index < n_folds:
            # Copy model and training data to keep each run independent
            cur_fold_model = copy.deepcopy(cur_model)
            cur_fold_train_data = copy.deepcopy(train_data)

            if to_gpu is True:
                cur_fold_model.cuda()

            if train_function != elasticnet_drp_train:
                # Must instantiate optimizer after putting model on the GPU
                cur_fold_optimizer = optim.AdamW(cur_fold_model.parameters(),
                                                 lr=learning_rate,
                                                 weight_decay=0.01,
                                                 amsgrad=True)
            else:
                cur_fold_optimizer = None

            if cv_resume is True and train_function != elasticnet_drp_train:
                cur_fold_model.load_state_dict(cur_checkpoint['model_state_dict'])
                cur_fold_optimizer.load_state_dict(cur_checkpoint['optimizer_state_dict'])
                early_stopper = checkpoint_early_stopper
            else:
                # Setup early stopping with no saving
                early_stopper = EarlyStopping(patience=patience, save=False, lower_better=True, verbose=verbose,
                                              delta=delta)

            # Check early_stopper for termination
            if early_stopper.early_stop:
                print("Early stopping reached for this CV index, skipping...")
                cv_index += 1
                cur_epoch = 1
                cv_resume = False
                continue

            cur_fold = cv_folds[cv_index]

            if omic_standardize is True:
                # NEW: Standardize training and validation data based on training data's statistics
                cur_fold_train_data.standardize(train_idx=cur_fold[0])

            # if train_function == elasticnet_drp_train:
            #     cur_fold_train_data.feature_selection(train_idx=cur_fold[0], method=f_regression, k=2000)

            cur_train_sampler = SubsetRandomSampler(cur_fold[0])
            cur_valid_sampler = SubsetRandomSampler(cur_fold[1])
            # Create data loaders based on current fold's indices
            train_loader = cur_data_loader(cur_fold_train_data, batch_size=batch_size,
                                           sampler=cur_train_sampler,
                                           num_workers=NUM_WORKERS, pin_memory=False, drop_last=True)
            # load validation data
            valid_loader = cur_data_loader(cur_fold_train_data, batch_size=batch_size,
                                           sampler=cur_valid_sampler,
                                           num_workers=NUM_WORKERS, pin_memory=False, drop_last=True)

            if verbose:
                print("Length of CV", str(cv_index), "train_loader:", str(len(train_loader)))
                print("Length of CV", str(cv_index), "valid_loader:", str(len(valid_loader)))

            if theoretical_loss is True:
                if epoch_save_folder is None:
                    exit("Epoch Save Folder not provided!")

                # Determine loss on validation set with the randomly initialized model
                _, valid_losses, random_results = train_function(valid_loader=valid_loader,
                                                                 cur_model=cur_model,
                                                                 criterion=criterion,
                                                                 optimizer=None,
                                                                 epoch=1,
                                                                 return_results=True,
                                                                 verbose=verbose)
                print("Average random model validation loss is:", valid_losses.avg)
                all_untrained_losses.append(valid_losses.avg)

                Path(epoch_save_folder).mkdir(parents=True, exist_ok=True)
                dest = epoch_save_folder + "/CV_Index_" + str(cv_index) + '_random_model_inference_results.csv'
                print("Saving epoch results to:", dest)
                random_results.to_csv(dest, float_format='%g')

                # Reset optimizer
                # cur_fold_optimizer = optim.Adam(cur_fold_model.parameters(), lr=learning_rate)

            # Do max_epochs steps;

            while cur_epoch < max_epochs + 1:
                # Train and validate for this epoch
                train_losses, valid_losses = train_function(train_loader=train_loader, valid_loader=valid_loader,
                                                            cur_model=cur_fold_model, criterion=criterion,
                                                            optimizer=cur_fold_optimizer,
                                                            epoch=cur_epoch, verbose=verbose)

                if save_epoch_results is True:
                    if epoch_save_folder is None:
                        exit("Epoch Save Folder not provided!")

                    # Save all training set predictions for the current epoch
                    temp_loader = cur_data_loader(cur_fold_train_data, batch_size=batch_size,
                                                  num_workers=NUM_WORKERS, pin_memory=False,
                                                  drop_last=True)
                    print("Getting current epoch predictions...")
                    _, valid_losses, all_results = train_function(valid_loader=temp_loader,
                                                                  cur_model=cur_fold_model, criterion=criterion,
                                                                  optimizer=cur_fold_optimizer,
                                                                  epoch=cur_epoch, verbose=False,
                                                                  return_results=True)

                    dest = epoch_save_folder + "/CV_Index_" + str(cv_index) + "_Epoch_" + \
                           str(cur_epoch) + '_inference_results.csv'
                    print("Saving epoch results to:", dest)
                    all_results.to_csv(dest, float_format='%g')

                # See if early stopping criteria are met
                early_stopper(score=valid_losses.avg, model=cur_fold_model, optimizer=cur_fold_optimizer,
                              epoch=cur_epoch)

                if verbose:
                    print("Epoch", str(cur_epoch), "CV:", str(cv_index),
                          "\n\t-> Average train losses:", str(train_losses.avg),
                          "\n\t-> Average valid losses:", str(valid_losses.avg))
                    print("Current best average validation loss:", early_stopper.best_score)

                if early_stopper.early_stop:
                    if verbose:
                        print("Stopping Training Early!")
                    all_avg_train_losses.append(train_losses.avg)
                    all_avg_valid_losses.append(early_stopper.best_score)
                    all_final_epochs.append(early_stopper.best_epoch)
                    if save_model is True:
                        if train_function == elasticnet_drp_train:
                            model_save(cv_index, cur_epoch, cur_fold_model, None, None, None,
                                       all_avg_train_losses, all_avg_valid_losses, save_model_path,
                                       save_model_frequency,
                                       early_stopper, all_final_epochs, sklearn=True, force=True)
                        else:
                            model_save(cv_index, cur_epoch, cur_fold_model, cur_fold_optimizer, train_losses,
                                       valid_losses,
                                       all_avg_train_losses, all_avg_valid_losses, save_model_path,
                                       save_model_frequency,
                                       early_stopper, all_final_epochs,
                                       force=True)
                    break

                elif cur_epoch == max_epochs:
                    # If we reach maximum allowed epochs
                    all_avg_train_losses.append(train_losses.avg)
                    all_avg_valid_losses.append(early_stopper.best_score)
                    all_final_epochs.append(cur_epoch)
                    if save_model is True:
                        if train_function == elasticnet_drp_train:
                            model_save(cv_index, cur_epoch, cur_fold_model, None, None, None,
                                       all_avg_train_losses, all_avg_valid_losses, save_model_path,
                                       save_model_frequency,
                                       early_stopper, all_final_epochs, sklearn=True, force=True)
                        else:
                            model_save(cv_index, cur_epoch, cur_fold_model, cur_fold_optimizer, train_losses,
                                       valid_losses,
                                       all_avg_train_losses, all_avg_valid_losses, save_model_path,
                                       save_model_frequency,
                                       early_stopper, all_final_epochs,
                                       force=True)

                else:
                    if save_model is True:
                        if train_function == elasticnet_drp_train:
                            model_save(cv_index, cur_epoch, cur_fold_model, None, None, None,
                                       all_avg_train_losses, all_avg_valid_losses, save_model_path,
                                       save_model_frequency,
                                       early_stopper, all_final_epochs, sklearn=True)
                        else:
                            model_save(cv_index, cur_epoch, cur_fold_model, cur_fold_optimizer, train_losses,
                                       valid_losses,
                                       all_avg_train_losses, all_avg_valid_losses, save_model_path,
                                       save_model_frequency,
                                       early_stopper, all_final_epochs)

                # Add to epoch counter
                cur_epoch += 1

            if verbose:
                print("CV", str(cv_index), ": Avg Train Loss ->", all_avg_train_losses,
                      ", Avg Valid Loss ->", all_avg_valid_losses)

            if save_cv_preds is True:
                print("Getting final CV", cv_index, "predictions...")
                _, valid_losses, all_results = train_function(valid_loader=valid_loader,
                                                              cur_model=cur_fold_model, criterion=criterion,
                                                              optimizer=cur_fold_optimizer,
                                                              epoch=cur_epoch, verbose=False,
                                                              return_results=True)

                dest = epoch_save_folder + "/CV_Index_" + str(cv_index) + "_Epoch_" + \
                       str(cur_epoch) + '_final_validation_results.csv'
                print("Saving epoch results to:", dest)
                all_results.to_csv(dest, float_format='%g')
            # The initial model (untrained) must be used for each fold setup
            del cur_fold_model
            del cur_fold_optimizer

            # Add to CV counter, reset epoch and checkpoint usage
            cv_index += 1
            cur_epoch = 1
            cv_resume = False

        # Return average of average (grand mean) of training and validation losses after running max_epochs
        avg_train_losses = sum(all_avg_train_losses) / len(all_avg_train_losses)
        avg_valid_losses = sum(all_avg_valid_losses) / len(all_avg_valid_losses)
        max_final_epoch = max(all_final_epochs)
        avg_final_epoch = sum(all_final_epochs) // len(all_final_epochs)  # Must be an integer

        avg_untrained_loss = None
        if theoretical_loss is True:
            avg_untrained_loss = sum(all_untrained_losses) / len(all_untrained_losses)

    if redo_validation is True:
        print("Re-running validation!")
        start_time = time.time()
        cv_checkpoints = glob.glob(save_model_path + '/checkpoint_cv_*')
        print("Found the following checkpoints:", cv_checkpoints)
        for checkpoint in cv_checkpoints:
            # Try to load checkpoint
            cur_cv_idx = int(re.findall(r'\d+', checkpoint)[0])
            if train_function != elasticnet_drp_train:
                print("Loading checkpoint from:", save_model_path + "/checkpoint_cv_" + str(cur_cv_idx) + ".pt")
                cur_checkpoint = torch.load(save_model_path + "/checkpoint_cv_" + str(cur_cv_idx) + ".pt")
                # Copy the fold's model and data (to avoid cross-contamination of model and standardization parameters)
                cur_fold_model = copy.deepcopy(cur_model)
                cur_fold_model.load_state_dict(cur_checkpoint['model_state_dict'])
                cur_epoch = cur_checkpoint['epoch']
                print("Loaded checkpoint with CV:", cur_cv_idx, "and epoch:", cur_epoch)
            else:
                print("Loading checkpoint from:", save_model_path + "/checkpoint_cv_" + str(cur_cv_idx) + ".joblib")
                cur_fold_model = load(save_model_path + "/checkpoint_cv_" + str(cur_cv_idx) + ".joblib")
                cur_epoch = None

            if to_gpu is True:
                cur_fold_model.cuda()

            cur_fold_train_data = copy.deepcopy(train_data)

            cur_fold = cv_folds[cur_cv_idx]
            cur_valid_sampler = SubsetRandomSampler(cur_fold[1])

            if omic_standardize is True:
                # NEW: Standardize training and validation data based on training data's statistics
                cur_fold_train_data.standardize(train_idx=cur_fold[0])

            # load validation data
            valid_loader = cur_data_loader(cur_fold_train_data, batch_size=batch_size,
                                           sampler=cur_valid_sampler,
                                           num_workers=NUM_WORKERS, pin_memory=False, drop_last=False)
            print("Getting final CV", cur_cv_idx, "predictions...")
            _, valid_losses, all_results = train_function(valid_loader=valid_loader,
                                                          cur_model=cur_fold_model, criterion=criterion,
                                                          optimizer=None,
                                                          epoch=cur_epoch, verbose=False,
                                                          return_results=True)
            name_tag = epoch_save_folder.split("CrossValidation/")[1]
            if "FullModel" in name_tag:
                tag = "FullModel"
            else:
                tag = "ResponseOnly"

            if train_function != elasticnet_drp_train:
                data_types = epoch_save_folder.split("Drugs")[1][1:]
                cv_folder = "/scratch/l/lstein/ftaj//CV_Results/" + \
                            "HyperOpt_DRP_" + tag + '_' + data_types + '_' + name_tag
                dest = cv_folder + "/CV_Index_" + str(cur_cv_idx) + "_Epoch_" + \
                       str(cur_epoch) + '_final_validation_results.csv'
            else:
                data_types = name_tag.split("Baseline_ElasticNet_")[1]
                cv_folder = "/scratch/l/lstein/ftaj//CV_Results/HyperOpt_DRP_ResponseOnly_" + data_types + "_Baseline_ElasticNet/"
                dest = cv_folder + "/CV_Index_" + str(cur_cv_idx) + "_final_validation_results.csv"
            print("Saving epoch results to:", dest)
            all_results.to_csv(dest, float_format='%g')
        print("Finished validation for each fold! Took", time.time() - start_time, "seconds")
        exit(0)

    if redo_interpretation is True:
        print("Re-running interpretation!")
        start_time = time.time()

        name_tag = epoch_save_folder.split("CrossValidation/")[1]
        if "FullModel" in name_tag:
            tag = "FullModel"
        else:
            tag = "ResponseOnly"

        cv_checkpoints = glob.glob(save_model_path + '/checkpoint_cv_*')

        print("Found the following checkpoints:", cv_checkpoints)
        if len(cv_checkpoints) == 0:
            sys.exit("No checkpoints to interpret (for this config)")
        for checkpoint in cv_checkpoints:
            # Try to load checkpoint
            cur_cv_idx = int(re.findall(r'\d+', checkpoint)[0])
            if train_function != elasticnet_drp_train:
                print("Loading checkpoint from:", save_model_path + "/checkpoint_cv_" + str(cur_cv_idx) + ".pt")
                cur_checkpoint = torch.load(save_model_path + "/checkpoint_cv_" + str(cur_cv_idx) + ".pt")
                # Copy the fold's model and data (to avoid cross-contamination of model and standardization parameters)
                cur_fold_model = copy.deepcopy(cur_model)
                cur_fold_model.load_state_dict(cur_checkpoint['model_state_dict'])

                cur_epoch = cur_checkpoint['epoch']
                print("Loaded checkpoint with CV:", cur_cv_idx, "and epoch:", cur_epoch)
            else:
                print("Loading checkpoint from:", save_model_path + "/checkpoint_cv_" + str(cur_cv_idx) + ".joblib")
                cur_fold_model = load(save_model_path + "/checkpoint_cv_" + str(cur_cv_idx) + ".joblib")
                cur_epoch = None

            if "LMF" in name_tag:
                print("Model uses LMF, turning to test mode...")
                # The following ensures compatibility with IntegratedGradients
                cur_fold_model.mode = "test"

            if to_gpu is True:
                cur_fold_model.cuda()

            cur_fold_train_data = copy.deepcopy(train_data)
            cur_fold = cv_folds[cur_cv_idx]

            final_subset_indices = cur_fold[1]
            if min_target is not None:
                # Get all the validation indices and respective AAC scores for those samples
                all_valid_aac = cur_fold_train_data.drug_data_targets
                # Find indices that satisfy condition
                subset_valid_aac = all_valid_aac >= min_target
                cur_subset_indices = torch.flatten(subset_valid_aac.nonzero())
                final_subset_indices = list(set(cur_subset_indices.int().tolist()) & set(final_subset_indices))
                final_subset_indices = torch.Tensor(final_subset_indices)

                # print("Subset indices for given minimum AAC target:", final_subset_indices)

            if dr_sub_cpd_names is not None:
                all_valid_drug_names = set(cur_fold_train_data.drug_names)
                subset_valid_drug_names = all_valid_drug_names.intersection(dr_sub_cpd_names)
                all_drug_names_np = np.array(cur_fold_train_data.drug_names, dtype=str)
                all_subset_indices = []
                for drug in subset_valid_drug_names:
                    # print("Drug in loop:", drug)
                    # cur_drug_indices = [i for i, x in enumerate(cur_fold_train_data.drug_names) if x == "drug"]
                    cur_drug_indices = np.where(all_drug_names_np == drug)[0].tolist()
                    all_subset_indices.append(cur_drug_indices)
                # flatten list
                all_subset_indices = [idx for subset in all_subset_indices for idx in subset]
                # all_subset_indices = torch.flatten(torch.Tensor(all_subset_indices))
                # find overlap with subset_indices
                final_subset_indices = list(set(final_subset_indices.int().tolist()) & set(all_subset_indices))
                final_subset_indices = torch.Tensor(final_subset_indices)
                # print("Subset indices for given compound names:", final_subset_indices)
                # subset_indices = np.intersect1d(subset_indices.cpu(), all_subset_indices)

            final_subset_indices = final_subset_indices.int()
            # print("Subset indices for given compound names:", final_subset_indices)

            if len(final_subset_indices) == 0:
                print("No matching samples in this fold, skipping...")
                continue

            cur_valid_sampler = SubsetRandomSampler(final_subset_indices)

            if omic_standardize is True:
                # Standardize training and validation data based on training data's statistics
                cur_fold_train_data.standardize(train_idx=cur_fold[0])

            if train_function == gnn_drp_train:
                # Must define a custom forward function that takes GNNs and works with IntegratedGradients
                def custom_forward(*inputs):
                    # omic_data, graph_x, graph_edge_attr, graph_edge_index, omic_length):
                    omic_length = inputs[-1]
                    omic_data = inputs[0:omic_length]
                    graph_x = inputs[-5]
                    graph_edge_attr = inputs[-4]
                    graph_edge_index = inputs[-3]
                    batch = inputs[-2]

                    return cur_fold_model([graph_x, graph_edge_index, graph_edge_attr, batch], omic_data)

                cur_interpret_method = IntegratedGradients(custom_forward)

            else:
                raise NotImplementedError
                # if interpret_method == "deeplift":
                #     cur_interpret_method = DeepLift(cur_model, multiply_by_inputs=False)
                # elif interpret_method == "deepliftshap":
                #     cur_interpret_method = DeepLiftShap(cur_model, multiply_by_inputs=False)
                # elif interpret_method == "integratedgradients":
                #     cur_interpret_method = IntegratedGradients(cur_model, multiply_by_inputs=False)
                # elif interpret_method == "ablation":
                #     cur_interpret_method = FeatureAblation(cur_model)

                # else:
                #     Warning("Incorrect interpretation method selected, defaulting to IntegratedGradients")
                #     cur_interpret_method = IntegratedGradients(cur_model, multiply_by_inputs=False)

            # Must put PairData class into test mode to get correct results
            cur_fold_train_data.mode = "test"
            # load validation data
            valid_loader = cur_data_loader(cur_fold_train_data, batch_size=1,
                                           sampler=cur_valid_sampler, shuffle=False,
                                           num_workers=0, pin_memory=False, drop_last=False)
            print("Number of batches to process:", str(len(valid_loader)))

            if train_function != elasticnet_drp_train:
                data_types = epoch_save_folder.split("Drugs")[1][1:]
                cv_folder = "/scratch/l/lstein/ftaj//CV_Results/" + \
                            "HyperOpt_DRP_" + tag + '_' + data_types + '_' + name_tag
                result_address = cv_folder + "/CV_Index_" + str(cur_cv_idx) + "_Epoch_" + \
                             str(cur_epoch) + '_final_interpretation_results.csv'
            else:
                data_types = name_tag.split("Baseline_ElasticNet_")[1]
                cv_folder = "/scratch/l/lstein/ftaj//CV_Results/HyperOpt_DRP_ResponseOnly_" + data_types + "_Baseline_ElasticNet/"
                result_address = cv_folder + "/CV_Index_" + str(cur_cv_idx) + "_final_interpretation_results.csv"

            data_types = data_types.split('_')
            omic_types = data_types[1:]
            all_interpret_results = []
            fold_start_time = time.time()
            for i, cur_samples in enumerate(valid_loader):

                if train_function == gnn_drp_train:
                    # PairData() output for GNN in test mode is not Batch(), so must reshape manually
                    cur_samples[1][0] = torch.squeeze(cur_samples[1][0], 0)
                    cur_samples[1][1] = torch.squeeze(cur_samples[1][1], 0)
                    cur_samples[1][2] = torch.squeeze(cur_samples[1][2], 0)
                    # Add batch
                    cur_samples[1] += [torch.zeros(cur_samples[1][0].shape[0], dtype=int).cuda()]

                    cur_output = cur_fold_model(cur_samples[1], cur_samples[2])
                    # Do not use LDS during inference
                    cur_loss = criterion(cur_output, cur_samples[3])
                    cur_loss = cur_loss.tolist()
                    # cur_loss = [loss[0] for loss in cur_loss]

                    cur_targets = cur_samples[3].tolist()
                    cur_targets = [target[0] for target in cur_targets]
                    cur_preds = cur_output.tolist()
                    # cur_preds = [pred[0] for pred in cur_preds]
                else:
                    cur_output = cur_fold_model(*cur_samples[1])
                    # Do not use LDS during inference
                    cur_loss = criterion(cur_output, cur_samples[2])
                    cur_loss = cur_loss.tolist()
                    cur_loss = [loss[0] for loss in cur_loss]

                    cur_targets = cur_samples[2].tolist()
                    cur_targets = [target[0] for target in cur_targets]
                    cur_preds = cur_output.tolist()
                    cur_preds = [pred[0] for pred in cur_preds]

                omic_length = len(cur_samples[2])

                zero_dl_attr_train, \
                zero_dl_delta_train = cur_interpret_method.attribute(
                    (*cur_samples[2], cur_samples[1][0], cur_samples[1][2]),
                    additional_forward_args=(
                        # cur_samples[1].x,
                        # cur_samples[1].edge_attr,
                        cur_samples[1][1],
                        cur_samples[1][3],
                        omic_length
                        # cur_samples[1].smiles[0]
                    ),
                    internal_batch_size=1,
                    return_convergence_delta=True)

                cur_dict = {'cpd_name': cur_samples[0]['drug_name'],
                            'cell_name': cur_samples[0]['cell_line_name'],
                            'target': cur_targets,
                            'predicted': cur_preds,
                            'RMSE_loss': cur_loss,
                            'interpret_delta': zero_dl_delta_train.tolist()
                            }

                # Recreate drug graph to use for interpretation
                cur_graph = MyGNNData(x=cur_samples[1][0], edge_index=cur_samples[1][1],
                                      edge_attr=cur_samples[1][2], smiles=cur_samples[0]['smiles'][0])
                cur_graph = GenFeatures()(cur_graph)

                # Create drug plotting directory if it doesn't exist
                Path(cv_folder + "/drug_plots/").mkdir(parents=True, exist_ok=True)
                # Plot positive and negative attributions on the drug molecule and save as a PNG plot
                drug_interpret_viz(edge_attr=zero_dl_attr_train[-1], node_attr=zero_dl_attr_train[-2],
                                   drug_graph=cur_graph, sample_info=cur_samples[0],
                                   plot_address=cv_folder + "/drug_plots/")

                for j in range(0, len(zero_dl_attr_train) - 2):  # ignore drug data (for now)
                    cur_col_names = cur_fold_train_data.omic_column_names[j]
                    for jj in range(len(cur_col_names)):
                        cur_dict[omic_types[j] + '_' + cur_col_names[jj]] = zero_dl_attr_train[j][:, jj].tolist()

                all_interpret_results.append(pd.DataFrame.from_dict(cur_dict))

                if i % 100 == 0:
                    print("Current batch:", i, ", elapsed:", str(time.time() - fold_start_time), "seconds")

            results_df = pd.concat(all_interpret_results)

            print("Saving epoch results to:", result_address)
            results_df.to_csv(result_address, float_format='%g')
            # fold_start_time = time.time()

        print("Finished interpretation for each fold! Took", time.time() - start_time, "seconds")
        exit(0)

    if final_full_train is False or train_function == elasticnet_drp_train:
        return avg_train_losses, avg_valid_losses, avg_untrained_loss, max_final_epoch

    else:
        cur_fold_model = copy.deepcopy(cur_model)
        cur_fold_train_data = copy.deepcopy(train_data)
        cur_epoch = 0
        if resume is True:
            print("Trying to load training checkpoint...")
            if save_model_path is None:
                exit("Model path must be given for resume!")
            # Check folder for training checkpoint
            checkpoints = glob.glob(save_model_path + '/checkpoint_train.pt')

            if len(checkpoints) == 0:
                print("Checkpoint not found! Starting from scratch...")
                resume = False

            else:
                print("Loading checkpoint from:", save_model_path + "/checkpoint_train.pt")
                # Load checkpoint and check for max epoch
                cur_checkpoint = torch.load(save_model_path + "/checkpoint_train.pt")
                cur_epoch = cur_checkpoint['epoch']
                avg_final_epoch = cur_checkpoint['avg_final_epoch']
                print("Loaded checkpoint at epoch:", cur_epoch)
                cur_fold_model.load_state_dict(cur_checkpoint['model_state_dict'])

        print("Training model on all training data!")
        if summary_writer is not None:
            print("Writing TensorBoard summaries too!")
            # for name, param in cur_fold_model.named_parameters():
            #     # Must force the retention of gradients for intermediate layers
            #     param.retain_grad()

        if to_gpu is True:
            cur_fold_model = cur_fold_model.cuda()

        # Must instantiate optimizer after putting model on the GPU
        cur_fold_optimizer = optim.AdamW(cur_fold_model.parameters(),
                                         lr=learning_rate,
                                         weight_decay=0.01,
                                         amsgrad=True)
        if resume is True:
            cur_fold_optimizer.load_state_dict(cur_checkpoint['optimizer_state_dict'])

        if omic_standardize is True:
            # NEW: Standardize all data (treated as training data)
            cur_fold_train_data.standardize()

        train_loader = cur_data_loader(cur_fold_train_data, batch_size=batch_size, shuffle=True,
                                       num_workers=NUM_WORKERS, pin_memory=False, drop_last=True)
        # for cur_epoch in range(avg_final_epoch):  # TODO Change to average final epoch?
        while cur_epoch < avg_final_epoch + 1:
            # Train and validate for this epoch
            train_losses = train_function(train_loader=train_loader,
                                          cur_model=cur_fold_model, criterion=criterion,
                                          optimizer=cur_fold_optimizer,
                                          epoch=cur_epoch + 1, verbose=verbose)
            if verbose:
                print("Epoch", str(cur_epoch + 1),
                      "\n-> Average train losses:", str(train_losses.avg))

            if save_model is True:
                if cur_epoch % save_model_frequency == 0:
                    print("Saving training model at epoch:", cur_epoch)
                    torch.save({
                        'epoch': cur_epoch + 1,  # +1 to not redo the same epoch on resume
                        'model_state_dict': cur_fold_model.state_dict(),
                        'optimizer_state_dict': cur_fold_optimizer.state_dict(),
                        'train_losses': train_losses,
                        'avg_final_epoch': avg_final_epoch
                    }, save_model_path + "/checkpoint_train.pt")

            if save_epoch_results:
                if epoch_save_folder is None:
                    Warning("Epoch Save Folder not provided, skipping save...")
                    continue

                # Save all training set predictions for the current epoch
                temp_loader = cur_data_loader(cur_fold_train_data, batch_size=batch_size,
                                              num_workers=NUM_WORKERS, pin_memory=False,
                                              drop_last=True)

                print("Getting current epoch predictions...")
                _, valid_losses, all_results = train_function(valid_loader=temp_loader,
                                                              cur_model=cur_fold_model, criterion=criterion,
                                                              optimizer=cur_fold_optimizer,
                                                              epoch=cur_epoch + 1, verbose=False,
                                                              return_results=True)
                dest = epoch_save_folder + "/TrainOnly" + "_Epoch_" + \
                       str(cur_epoch) + '_inference_results.csv'
                print("Saving epoch results to:", dest)
                all_results.to_csv(dest, float_format='%g')

            if summary_writer is not None:
                summary_writer.add_scalar('Loss', train_losses.avg, cur_epoch + 1)

                summary_writer.add_histogram("drp_module[0].drp_0_linear.bias",
                                             cur_fold_model.drp_module[0].custom_dense.drp_0_linear.bias, cur_epoch + 1)
                summary_writer.add_histogram("drp_module[0].drp_0_linear.weight",
                                             cur_fold_model.drp_module[0].custom_dense.drp_0_linear.weight,
                                             cur_epoch + 1)
                # summary_writer.add_histogram("drp_module[0].drp_0_linear.activation", cur_fold_model.drp_module[0].custom_dense.drp_0_activation, cur_epoch + 1)
                summary_writer.add_histogram("drp_module[0].drp_0_linear.weight.grad",
                                             cur_fold_model.drp_module[0].custom_dense.drp_0_linear.weight.grad,
                                             cur_epoch + 1)

                summary_writer.add_histogram("drp_module[1].drp_1_linear.bias",
                                             cur_fold_model.drp_module[1].custom_dense.drp_1_linear.bias, cur_epoch + 1)
                summary_writer.add_histogram("drp_module[1].drp_1_linear.weight",
                                             cur_fold_model.drp_module[1].custom_dense.drp_1_linear.weight,
                                             cur_epoch + 1)
                # summary_writer.add_histogram("drp_module[1].drp_1_linear.activation", cur_fold_model.drp_module[1].custom_dense.drp_1_activation, cur_epoch + 1)
                summary_writer.add_histogram("drp_module[1].drp_1_linear.weight.grad",
                                             cur_fold_model.drp_module[1].custom_dense.drp_1_linear.weight.grad,
                                             cur_epoch + 1)

                summary_writer.add_histogram("drp_module[2].drp_2_linear.bias",
                                             cur_fold_model.drp_module[2].custom_dense.drp_2_linear.bias, cur_epoch + 1)
                summary_writer.add_histogram("drp_module[2].drp_2_linear.weight",
                                             cur_fold_model.drp_module[2].custom_dense.drp_2_linear.weight,
                                             cur_epoch + 1)
                # summary_writer.add_histogram("drp_module[2].drp_2_linear.activation", cur_fold_model.drp_module[2].custom_dense.drp_2_activation, cur_epoch + 1)
                summary_writer.add_histogram("drp_module[2].drp_2_linear.weight.grad",
                                             cur_fold_model.drp_module[2].custom_dense.drp_2_linear.weight.grad,
                                             cur_epoch + 1)

                summary_writer.add_histogram("drp_module[3].drp_3_linear.bias",
                                             cur_fold_model.drp_module[3].custom_dense.drp_3_linear.bias, cur_epoch + 1)
                summary_writer.add_histogram("drp_module[3].drp_3_linear.weight",
                                             cur_fold_model.drp_module[3].custom_dense.drp_3_linear.weight,
                                             cur_epoch + 1)
                # summary_writer.add_histogram("drp_module[3].drp_3_linear.activation", cur_fold_model.drp_module[3].custom_dense.drp_3_activation, cur_epoch + 1)
                summary_writer.add_histogram("drp_module[3].drp_3_linear.weight.grad",
                                             cur_fold_model.drp_module[3].custom_dense.drp_3_linear.weight.grad,
                                             cur_epoch + 1)

                summary_writer.add_histogram("drp_module[4].drp_4_linear.bias",
                                             cur_fold_model.drp_module[4].custom_dense.drp_4_linear.bias, cur_epoch + 1)
                summary_writer.add_histogram("drp_module[4].drp_4_linear.weight",
                                             cur_fold_model.drp_module[4].custom_dense.drp_4_linear.weight,
                                             cur_epoch + 1)
                # summary_writer.add_histogram("drp_module[4].drp_4_linear.activation", cur_fold_model.drp_module[4].custom_dense.drp_4_activation, cur_epoch + 1)
                summary_writer.add_histogram("drp_module[4].drp_4_linear.weight.grad",
                                             cur_fold_model.drp_module[4].custom_dense.drp_4_linear.weight.grad,
                                             cur_epoch + 1)

                summary_writer.add_histogram("cur_fold_model.encoders[0].lin2.bias",
                                             cur_fold_model.encoders[0].lin2.bias, cur_epoch + 1)
                summary_writer.add_histogram("cur_fold_model.encoders[0].lin2.weight",
                                             cur_fold_model.encoders[0].lin2.weight, cur_epoch + 1)
                summary_writer.add_histogram("cur_fold_model.encoders[0].lin2.weight.grad",
                                             cur_fold_model.encoders[0].lin2.weight.grad, cur_epoch + 1)
                # summary_writer.add_histogram("drp_module[4].drp_4_linear.bias", cur_fold_model.encoders[1].autoencoder.encoder.coder[0].custom_dense.drp_4_linear.bias, cur_epoch + 1)
                # summary_writer.add_histogram("drp_module[4].drp_4_linear.weight", cur_fold_model.drp_module[4].custom_dense.drp_4_linear.weight, cur_epoch + 1)

                summary_writer.add_histogram(
                    "cur_fold_model.encoders[1].encoder.coder[1].custom_dense.code_layer__linear.bias",
                    cur_fold_model.encoders[1].encoder.coder[1].custom_dense.code_layer__linear.bias, cur_epoch + 1)
                summary_writer.add_histogram(
                    "cur_fold_model.encoders[1].encoder.coder[1].custom_dense.code_layer__linear.weight",
                    cur_fold_model.encoders[1].encoder.coder[1].custom_dense.code_layer__linear.weight, cur_epoch + 1)
                summary_writer.add_histogram(
                    "cur_fold_model.encoders[1].encoder.coder[1].custom_dense.code_layer__linear.weight.grad",
                    cur_fold_model.encoders[1].encoder.coder[1].custom_dense.code_layer__linear.weight.grad,
                    cur_epoch + 1)
            # update cur_epoch counter
            cur_epoch += 1

        return cur_fold_model, avg_train_losses, avg_valid_losses, avg_untrained_loss, max_final_epoch
