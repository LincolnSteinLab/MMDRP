import copy
import time

import torch
from torch import optim
from torch.utils import data
from torch.utils.data import SubsetRandomSampler

from CustomFunctions import AverageMeter, EarlyStopping
from DataImportModules import AutoEncoderPrefetcher, DataPrefetcher


def morgan_train(cur_model, criterion, optimizer, epoch, train_loader=None, valid_loader=None, verbose: bool = False):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    train_losses = AverageMeter('Training Loss', ':.4e')
    valid_losses = AverageMeter('Validation Loss', ':.4e')

    prefetcher = AutoEncoderPrefetcher(train_loader)
    cur_data = prefetcher.next()
    # Training
    cur_model.train()
    end = time.time()
    data_end_time = time.time()

    if verbose:
        print("Training Drug Model...")
    i = 0
    # running_loss = 0.0
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
        cur_data = prefetcher.next()

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
        prefetcher = AutoEncoderPrefetcher(valid_loader)
        cur_data = prefetcher.next()
        # Training
        cur_model.eval()
        end = time.time()
        data_end_time = time.time()

        if verbose:
            print("Validating Drug Model...")
        i = 0
        # running_loss = 0.0
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
                # running_loss += valid_loss.item()
                # if verbose:
                #     if i % 200 == 199:  # print every 200 mini-batches
                #         print('[%d, %5d] valid loss: %.3f' %
                #               (epoch, i, running_loss / 200))
                #         running_loss = 0.0

                data_end_time = time.time()
                cur_data = prefetcher.next()

    if verbose:
        print("Sum train and valid losses at epoch", epoch, ":", train_losses.sum, valid_losses.sum)

    return train_losses, valid_losses


def omic_train(cur_model, criterion, optimizer, epoch, train_loader=None, valid_loader=None,
               verbose: bool = False):
    batch_time = AverageMeter('Time', ':6.3f')
    # data_time = AverageMeter('Data', ':6.3f')
    train_losses = AverageMeter('Training Loss', ':.4e')
    valid_losses = AverageMeter('Validation Loss', ':.4e')

    if train_loader is not None:
        # === Train Model ===
        prefetcher = AutoEncoderPrefetcher(train_loader)
        cur_data = prefetcher.next()
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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # data_end_time = time.time()
            cur_data = prefetcher.next()

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
        prefetcher = AutoEncoderPrefetcher(valid_loader)
        cur_data = prefetcher.next()
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
                cur_data = prefetcher.next()

                # running_loss += valid_loss.item()

                # if i % 200 == 199:  # print every 200 mini-batches
                #     print('[%d, %5d] valid loss: %.3f' %
                #           (epoch, i, running_loss / 200))
                #     running_loss = 0.0

    if verbose:
        print("Sum train and valid losses at epoch", epoch, ":", train_losses.sum, valid_losses.sum)

    return train_losses, valid_losses


def drp_train(cur_model, criterion, optimizer, epoch, train_loader=None, valid_loader=None, verbose: bool = False):
    data_time = AverageMeter('Data', ':6.3f')
    train_losses = AverageMeter('Training Loss', ':.4e')
    valid_losses = AverageMeter('Validation Loss', ':.4e')
    if train_loader is None and valid_loader is None:
        raise AssertionError("At least one of train loader or valid loader must be provided!")

    if train_loader is not None:
        # ==== Train Model ====
        # switch to train mode
        cur_model.train()
        end = time.time()
        data_end_time = time.time()

        prefetcher = DataPrefetcher(train_loader)
        _, cur_dr_data, cur_dr_target = prefetcher.next()
        # Check data format once before training
        assert len(cur_dr_data) == cur_model.len_encoder_list, "DataPrefetcher DR data doesn't match # encoders"

        if verbose:
            print("Training DRP Model...")
        # with profiler.profile(record_shapes=False, use_cuda=True, profile_memory=True) as prof:
        #   with profiler.record_function("model_inference"):
        i = 0
        running_loss = 0.0
        while cur_dr_data is not None:
            i += 1

            # print("Data generation time for batch", i, ":", time.time() - data_end_time)
            data_time.update(time.time() - data_end_time)

            # forward + backward + optimize
            output = cur_model(*cur_dr_data)
            loss = criterion(output, cur_dr_target)

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
            train_losses.update(loss.item(), cur_dr_target.shape[0])
            running_loss += loss.item()
            if verbose:
                if i % 1000 == 999:  # print every 2000 mini-batches
                    print('[%d, %5d] train loss: %.6f' %
                          (epoch, i, running_loss / 1000))
                    running_loss = 0.0

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
            _, cur_dr_data, cur_dr_target = prefetcher.next()

    if valid_loader is None:
        return train_losses

    else:
        # ==== Validate Model ====
        # switch to evaluation mode
        cur_model.eval()
        prefetcher = DataPrefetcher(valid_loader)
        _, cur_dr_data, cur_dr_target = prefetcher.next()
        assert len(cur_dr_data) == cur_model.len_encoder_list, "DataPrefetcher DR data doesn't match # encoders"

        if verbose:
            print("Validating DRP Model...")
        # with profiler.profile(record_shapes=False, use_cuda=True, profile_memory=True) as prof:
        #   with profiler.record_function("model_inference"):
        i = 0
        running_loss = 0.0
        with torch.no_grad():
            # batch_start = time.time()
            # end = time.time()
            while cur_dr_data is not None:
                i += 1

                # forward + loss measurement
                output = cur_model(*cur_dr_data)
                valid_loss = criterion(output, cur_dr_target)
                # Second argument is effectively the batch size
                valid_losses.update(valid_loss.item(), cur_dr_target.shape[0])

                _, cur_dr_data, cur_dr_target = prefetcher.next()
                running_loss += valid_loss.item()
                if verbose:
                    if i % 200 == 199:  # print every 2000 mini-batches
                        print('[%d, %5d] valid loss: %.6f' %
                              (epoch, i, running_loss / 200))
                        running_loss = 0.0
        if verbose:
            print("Sum train and valid losses at epoch", epoch, ":", train_losses.sum, valid_losses.sum)

        return train_losses, valid_losses


def cross_validate(train_data, train_function, cv_folds, batch_size: int, cur_model, criterion,
                   #optimizer,
                   patience: int = 5, delta: float = 0.01, max_epochs: int = 100, learning_rate: float = 0.001,
                   final_full_train: bool = False, NUM_WORKERS: int = 0, theoretical_loss: bool = False,
                   verbose: bool = False):
    """
    This function will train the given model on len(cv_folds)-1 folds for max_epochs and then validates
    on the remaining fold.
    :cv_folds: a list of lists of format [[train_indices, valid_indices], ...] format
    :return: sum and average of train and validation losses for the current k-fold setup
    """
    n_folds = len(cv_folds)
    all_avg_train_losses = []
    all_avg_valid_losses = []
    all_final_epochs = []
    if verbose:
        print("Training + Validation Data Length:", str(len(train_data)))

    for cv_index in range(n_folds):
        cur_fold_model = copy.deepcopy(cur_model)
        cur_fold_model.cuda()
        cur_fold_optimizer = optim.Adam(cur_fold_model.parameters(), lr=learning_rate)
        # cur_fold_optimizer.cuda()

        cur_fold = cv_folds[cv_index]

        cur_train_sampler = SubsetRandomSampler(cur_fold[0])
        cur_valid_sampler = SubsetRandomSampler(cur_fold[1])
        # Create data loaders based on current fold's indices
        train_loader = data.DataLoader(train_data, batch_size=batch_size,
                                       sampler=cur_train_sampler,
                                       num_workers=NUM_WORKERS, pin_memory=False, drop_last=True)
        # load validation data in batches 4 times the size
        valid_loader = data.DataLoader(train_data, batch_size=batch_size,
                                       sampler=cur_valid_sampler,
                                       num_workers=NUM_WORKERS, pin_memory=False, drop_last=True)

        if verbose:
            print("Length of CV", str(cv_index), "train_loader:", str(len(train_loader)))
            print("Length of CV", str(cv_index), "valid_loader:", str(len(valid_loader)))

        if theoretical_loss is True:
            # Determine loss on validation set with the randomly initialized model
            _, valid_losses = train_function(valid_loader=valid_loader,
                                             cur_model=cur_fold_model,
                                             criterion=criterion,
                                             optimizer=None,
                                             epoch=1, verbose=verbose)
            print("Average random model validation loss is:", valid_losses.avg)

            # Reset optimizer
            # cur_fold_optimizer = optim.Adam(cur_fold_model.parameters(), lr=learning_rate)


        # Setup early stopping with no saving
        early_stopper = EarlyStopping(patience=patience, save=False, lower_better=True, verbose=verbose, delta=delta)
        # Do max_epochs steps;
        for cur_epoch in range(max_epochs):
            # Train and validate for this epoch
            train_losses, valid_losses = train_function(train_loader=train_loader, valid_loader=valid_loader,
                                                        cur_model=cur_fold_model, criterion=criterion,
                                                        optimizer=cur_fold_optimizer,
                                                        epoch=cur_epoch + 1, verbose=verbose)

            # See if early stopping criteria are met
            early_stopper(score=valid_losses.avg, model=cur_fold_model, optimizer=cur_fold_optimizer,
                          epoch=cur_epoch + 1)

            if verbose:
                print("Epoch", str(cur_epoch + 1), "CV:", str(cv_index),
                      "\n-> Average train losses:", str(train_losses.avg),
                      "\n-> Average valid losses:", str(valid_losses.avg))
                print("Current best average validation loss:", early_stopper.best_score)

            if early_stopper.early_stop:
                if verbose:
                    print("Stopping Training Early!")
                all_avg_train_losses.append(train_losses.avg)
                all_avg_valid_losses.append(early_stopper.best_score)
                all_final_epochs.append(early_stopper.best_epoch)
                break

            else:
                # If we reach maximum allowed epochs
                if cur_epoch + 1 == max_epochs:
                    all_avg_train_losses.append(train_losses.avg)
                    all_avg_valid_losses.append(early_stopper.best_score)
                    all_final_epochs.append(cur_epoch + 1)

        if verbose:
            print("CV", str(cv_index), ": Avg Train Loss ->", all_avg_train_losses,
                  ", Avg Valid Loss ->", all_avg_valid_losses)

        # The initial model (untrained) must be used for each fold setup
        del cur_fold_model
        del cur_fold_optimizer

    # Return average of average (grand mean) of training and validation losses after running max_epochs
    avg_train_losses = sum(all_avg_train_losses) / len(all_avg_train_losses)
    avg_valid_losses = sum(all_avg_valid_losses) / len(all_avg_valid_losses)
    max_final_epoch = max(all_final_epochs)

    if final_full_train is False:
        return avg_train_losses, avg_valid_losses, max_final_epoch

    else:
        # Train the model on all available data
        print("Training model on all training data!")
        cur_fold_model = copy.deepcopy(cur_model)
        cur_fold_optimizer = optim.Adam(cur_fold_model.parameters(), lr=learning_rate)
        train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                       num_workers=NUM_WORKERS, pin_memory=False, drop_last=True)
        for cur_epoch in range(max_final_epoch):
            # Train and validate for this epoch
            train_losses = train_function(train_loader=train_loader,
                                          cur_model=cur_fold_model, criterion=criterion,
                                          optimizer=cur_fold_optimizer,
                                          epoch=cur_epoch + 1, verbose=verbose)
            if verbose:
                print("Epoch", str(cur_epoch + 1),
                      "\n-> Average train losses:", str(train_losses.avg))

    return cur_fold_model, avg_train_losses, avg_valid_losses, max_final_epoch
