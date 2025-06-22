import argparse
import os
import statistics

import torch
import torch.nn as nn
import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.tools import multi_gpu_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Test dataset dir')
    parser.add_argument("--half", action='store_true',
                        help="whether train with half precision.")
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()

    multi_gpu_utils.init_distributed_mode(opt)

    print('-----------------Dataset Building------------------')
    # if we want to train from last checkpoint.
    if opt.model_dir:
        model_path = os.path.join(opt.model_dir, 'config.yaml')
        hypes = yaml_utils.load_yaml(model_path, None)
        opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
        opencood_validate_dataset = build_dataset(hypes, visualize=False, train=False)
        train_loader = DataLoader(opencood_train_dataset,
                                  # batch_size=hypes['train_params']['batch_size'],
                                  batch_size=1,
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch_train,
                                  shuffle=True,
                                  pin_memory=False,
                                  drop_last=True)
        val_loader = DataLoader(opencood_validate_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                shuffle=False,
                                pin_memory=False,
                                drop_last=True)
        print('---------------Creating Model------------------')
        model = train_utils.create_model(hypes)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path,
                                                         model)


    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
    model_without_ddp = model

    if opt.distributed:
        model = \
            torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[opt.gpu],
                                                      find_unused_parameters=True)
        model_without_ddp = model.module

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)
    # lr scheduler setup
    num_steps = len(train_loader)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps)

    # record training
    writer = SummaryWriter(saved_path)

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    print('Training start')
    epoches = hypes['train_params']['epoches']
    # used to help schedule learning rate

    # load rq2 select data
    train_data_number = 0
    augment_number = 996
    select_data_indices = []
    save_path = os.path.join(opt.dataset_dir, "rq2/select_indices.txt")
    with open(save_path, 'r') as file:
        for line in file:
            select_data_indices.append(int(line.rstrip()) + train_data_number)

    # intemediate only!
    params_path = os.path.join(opt.dataset_dir, "rq2/augment_params.txt")
    lossy_p = []
    chlossy_p = []
    with open(params_path, 'r') as file:
        count = 0
        l_flag = False
        chl_flag = False
        for line in file:
            if line.rstrip() == 'chlossy_p':
                chl_flag = True
                continue
            elif chl_flag == True and count < augment_number:
                chlossy_p.append(float(line.rstrip()))
                count += 1
            elif line.rstrip() == 'lossy_p':
                chl_flag = False
                l_flag = True
                count = 0
                continue
            elif l_flag == True:
                if count < augment_number:
                    lossy_p.append(float(line.rstrip()))
                    count += 1

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        print("loop start!")
        if hypes['lr_scheduler']['core_method'] != 'cosineannealwarm':
            scheduler.step(epoch)
        if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
            scheduler.step_update(epoch * num_steps + 0)
        for param_group in optimizer.param_groups:
            print('learning rate %.7f' % param_group["lr"])

        # if opt.distributed:
        #     sampler_train.set_epoch(epoch)

        pbar2 = tqdm.tqdm(total=len(train_loader), leave=True)

        for i, batch_data in enumerate(train_loader):
            # the model will be evaluation mode during validation
            if i not in select_data_indices and i >= train_data_number:
                continue
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            batch_data = train_utils.to_device(batch_data, device)

            # case1 : late fusion train --> only ego needed,
            # and ego is random selected
            # case2 : early fusion train --> all data projected to ego
            # case3 : intermediate fusion --> ['ego']['processed_lidar']
            # becomes a list, which containing all data from other cavs
            # as well
            if not opt.half:
                # train for intermediate models
                if i >= train_data_number + augment_number and \
                        i < train_data_number + augment_number * 2:
                    index = i - train_data_number - augment_number
                    batch_data['ego']['augment_op'] = 'chlossy'
                    batch_data['ego']['p'] = chlossy_p[index]
                elif i >= train_data_number + augment_number * 4 and \
                        i < train_data_number + augment_number * 5:
                    index = i - train_data_number - augment_number * 4
                    batch_data['ego']['augment_op'] = 'lossy'
                    batch_data['ego']['p'] = lossy_p[index]

                ouput_dict = model(batch_data['ego'])
                # first argument is always your output dictionary,
                # second argument is always your label dictionary.
                final_loss = criterion(ouput_dict,
                                       batch_data['ego']['label_dict'])
            else:
                with torch.cuda.amp.autocast():
                    ouput_dict = model(batch_data['ego'])
                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])

            criterion.logging(epoch, i, len(train_loader), writer, pbar=pbar2)
            pbar2.update(1)

            if not opt.half:
                with torch.autograd.set_detect_anomaly(True):
                    final_loss.backward()
                    optimizer.step()
            else:
                scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
                scheduler.step_update(epoch * num_steps + i)

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model_without_ddp.state_dict(),
                       os.path.join(saved_path, 'net_epoch%d.pth' % (epoch + 1)))

        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    ouput_dict = model(batch_data['ego'])

                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                    valid_ave_loss.append(final_loss.item())
            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                              valid_ave_loss))
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

    print('Training Finished, checkpoints saved to %s' % saved_path)


if __name__ == '__main__':
    main()
