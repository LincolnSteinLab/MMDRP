import time
import torch
from CustomFunctions import AverageMeter
from DataImportModules import AutoEncoderPrefetcher, DataPrefetcher


def morgan_train(train_loader, cur_model, criterion, optimizer, epoch, valid_loader=None):
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

    print("Training Drug Model...")
    i = 0
    running_loss = 0.0
    while cur_data is not None:
        i += 1
        data_time.update(time.time() - data_end_time)

        # forward + backward + optimize
        output = cur_model(cur_data)
        loss = criterion(output, cur_data)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        train_losses.update(loss.item(), cur_data.shape[0])

        data_end_time = time.time()
        cur_data = prefetcher.next()

        running_loss += loss.item()
        if i % 1000 == 999:    # print every 1000 mini-batches
            print('[%d, %5d] train loss: %.3f' %
                  (epoch, i, running_loss / 1000))
            running_loss = 0.0

    if valid_loader is None:
        return train_losses
    else:
        prefetcher = AutoEncoderPrefetcher(valid_loader)
        cur_data = prefetcher.next()
        # Training
        cur_model.eval()
        end = time.time()
        data_end_time = time.time()

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
                running_loss += valid_loss.item()
                if i % 200 == 199:    # print every 200 mini-batches
                    print('[%d, %5d] valid loss: %.3f' %
                          (epoch, i, running_loss / 200))
                    running_loss = 0.0

                data_end_time = time.time()
                cur_data = prefetcher.next()

    print("Sum train and valid losses at epoch", epoch, ":", train_losses.sum, valid_losses.sum)
    return train_losses, valid_losses


def omic_train(train_loader, cur_model, criterion, optimizer, epoch, valid_loader=None):
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

    print("Training Omic Model...")
    i = 0
    running_loss = 0.0
    while cur_data is not None:
        i += 1
        data_time.update(time.time() - data_end_time)

        # forward + backward + optimize
        output = cur_model(cur_data)
        loss = criterion(output, cur_data)

        optimizer.zero_grad(set_to_none=True)

        train_losses.update(loss.item(), cur_data.shape[0])
        loss.backward()
        optimizer.step()

        data_end_time = time.time()
        cur_data = prefetcher.next()

        running_loss += loss.item()
        if i % 1000 == 999:    # print every 1000 mini-batches
            print('[%d, %5d] train loss: %.3f' %
                  (epoch, i, running_loss / 1000))
            running_loss = 0.0

    if valid_loader is None:
        return train_losses
    else:
        prefetcher = AutoEncoderPrefetcher(valid_loader)
        cur_data = prefetcher.next()
        # Training
        cur_model.eval()
        end = time.time()
        data_end_time = time.time()

        print("Validating Omic Model...")
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
                running_loss += valid_loss.item()

                data_end_time = time.time()
                cur_data = prefetcher.next()

                if i % 200 == 199:    # print every 200 mini-batches
                    print('[%d, %5d] valid loss: %.3f' %
                          (epoch, i, running_loss / 200))
                    running_loss = 0.0

    print("Sum train and valid losses at epoch", epoch, ":", train_losses.sum, valid_losses.sum)
    return train_losses, valid_losses


def drp_train(train_loader, valid_loader, cur_model, criterion, optimizer, epoch, train_len, valid_len, batch_size):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    train_losses = AverageMeter('Training Loss', ':.4e')
    valid_losses = AverageMeter('Validation Loss', ':.4e')

    # TODO Uncomment
    # train_progress = ProgressMeter(
    #     train_len // batch_size,
    #     [batch_time, data_time, train_losses],
    #     prefix="Training Epoch: [{}]".format(epoch))
    # valid_progress = ProgressMeter(
    #     valid_len // (batch_size * 8),
    #     [batch_time, data_time, valid_losses],
    #     prefix="Validating Epoch: [{}]".format(epoch))

    # ==== Train Model ====
    # switch to train mode
    cur_model.train()
    end = time.time()
    data_end_time = time.time()

    prefetcher = DataPrefetcher(train_loader)
    _, cur_dr_data, cur_dr_target = prefetcher.next()
    assert len(cur_dr_data) == cur_model.len_encoder_list, "DataPrefetcher DR data doesn't match # encoders"

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
        # print("Model summary is:")
        # summary(cur_model, input_data=cur_dr_data)
        # exit()

        output = cur_model(*cur_dr_data)
        loss = criterion(output, cur_dr_target)

        # By default, gradients are accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called, so must reset to zero manually if needed.
        optimizer.zero_grad(set_to_none=True)

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
        if i % 1000 == 999:    # print every 2000 mini-batches
            print('[%d, %5d] train loss: %.3f' %
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

    # print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=30))
    # print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=30))
    # prof.export_chrome_trace("chrome_trace.json")
    # exit()
    # return train_losses, valid_losses

    # ==== Validate Model ====
    # switch to evaluation mode
    cur_model.eval()
    prefetcher = DataPrefetcher(valid_loader)
    _, cur_dr_data, cur_dr_target = prefetcher.next()
    assert len(cur_dr_data) == cur_model.len_encoder_list, "DataPrefetcher DR data doesn't match # encoders"

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
            if i % 200 == 199:    # print every 2000 mini-batches
                print('[%d, %5d] valid loss: %.3f' %
                      (epoch, i, running_loss / 200))
                running_loss = 0.0

    print("Sum train and valid losses at epoch", epoch, ":", train_losses.sum, valid_losses.sum)
    return train_losses, valid_losses

