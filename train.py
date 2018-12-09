from argparse import ArgumentParser
import os
from pathlib import Path

import numpy as np
import torch
import torch.utils.data

import data_utils
import models
from metrics import TimeMeter, MeanValue, Accuracy


class LRManager:

    def __init__(self, boundaries, values):
        self.boundaries = boundaries
        self.values = values

    def get(self, epoch):
        for b, v in zip(self.boundaries, self.values):
            if epoch < b:
                return v
        return self.values[-1]

    def set_lr_for_optim(self, epoch, optim):
        lr = self.get(epoch)
        for group in optim.param_groups:
            group['lr'] = lr


def init_param(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)


def make_param_groups(net, weight_decay):
    weight_params = []
    bias_params = []
    for name, value in net.named_parameters():
        if 'bias' in name:
            bias_params.append(value)
        else:
            weight_params.append(value)
    param_groups = [
        {'params': weight_params, 'weight_decay': weight_decay},
        {'params': bias_params, 'weight_decay': 0.}
    ]
    return param_groups


def main(args):

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load data
    data, meta = data_utils.load_data(
        args.dataset_root, args.dataset, is_training=True)
    train_data, val_data = data_utils.split_data(
        data, args.validate_rate, shuffle=True)

    # build train dataloader
    train_dataset = data_utils.ImageDataset(
        *train_data, is_training=True, is_flip=args.dataset not in ['mnist', 'svhn'])
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # build val dataloader
    val_dataset = data_utils.ImageDataset(*val_data, is_training=False)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # remove temp dataset variables to reduce memory usage
    del data, train_data, val_data

    device = torch.device(args.device)

    # build model
    if args.model == 'resnet_20':
        model = models.Resnet20
    else:
        model = models.SimpleCNN
    net = model(train_dataset.shape, meta['n_class']).to(device=device)
    net.apply(init_param)

    criterion = torch.nn.CrossEntropyLoss()

    # build optim
    optim = torch.optim.SGD(make_param_groups(
        net, args.weight_decay), 0.1, momentum=0.9)

    # make log directory
    logdir = Path(args.logdir)
    if not logdir.exists():
        os.makedirs(str(logdir))

    global_step = 0
    start_epoch = 0
    if args.restore:
        # restore checkpoint
        state = torch.load(args.restore)
        start_epoch = state['epoch'] + 1
        global_step = state['global_step']
        net.load_state_dict(state['net'])
        optim.load_state_dict(state['optim'])

    # lr strategy
    lr_boundaries = list(map(int, args.boundaries.split(',')))
    lr_values = list(map(float, args.values.split(',')))
    lr_manager = LRManager(lr_boundaries, lr_values)

    for e in range(start_epoch, args.n_epoch):
        print('-------epoch: {:d}-------'.format(e))

        # training phrase
        net.train()
        mean_loss, acc = MeanValue(), Accuracy()
        lr_manager.set_lr_for_optim(e, optim)
        tm = TimeMeter()
        tm.start()
        train_log = {}
        for i, (x, y) in enumerate(train_dataloader):
            tm.add_counter()

            if device.type == 'cuda':
                x = x.cuda(device, non_blocking=True)
                y = y.cuda(device, non_blocking=True)

            optim.zero_grad()
            logits = net(x)
            loss = criterion(logits, y)

            loss.backward()
            optim.step()
            global_step += 1

            loss = loss.detach().cpu().numpy()
            predicts = torch.argmax(logits, dim=1).detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            mean_loss.add(loss)
            acc.add(predicts, y)

            if i % args.log_every == 0:
                torch.cuda.synchronize()
                tm.stop()

                print('step: {:d}, lr: {:g}, loss: {:.4f}, acc: {:.2%}, speed: {:.2f} i/s.'
                      .format(i, lr_manager.get(e), mean_loss.get(), acc.get(), args.batch_size / tm.get()))
                train_log[global_step] = {
                    'loss': mean_loss.get(), 'acc': acc.get()}
                tm.reset()
                tm.start()
                mean_loss.reset()
                acc.reset()

        # val phrase
        net.eval()
        mean_loss, acc = MeanValue(), Accuracy()
        for x, y in val_dataloader:

            if device.type == 'cuda':
                x = x.cuda(device, non_blocking=True)
                y = y.cuda(device, non_blocking=True)

            logits = net(x)
            loss = criterion(logits, y)

            loss = loss.detach().cpu().numpy()
            predicts = torch.argmax(logits, dim=1).detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            mean_loss.add(loss)
            acc.add(predicts, y)

        print('val_loss: {:.4f}, val_acc: {:.2%}'.format(
            mean_loss.get(), acc.get()))
        val_log = {global_step: {'loss': mean_loss.get(), 'acc': acc.get()}}

        # save checkpoint
        vars_to_saver = {
            'net': net.state_dict(), 'optim': optim.state_dict(),
            'epoch': e, 'global_step': global_step}
        cpt_file = logdir / 'checkpoint_{:d}.pk'.format(e)
        torch.save(vars_to_saver, str(cpt_file))

        log_file = logdir / 'log_{:d}.pk'.format(e)
        torch.save({'train': train_log, 'val': val_log}, str(log_file))


if __name__ == '__main__':
    parser = ArgumentParser(description='pytorch for small image dataset')
    parser.add_argument('--dataset', default='cifar10',
                        help='the training dataset')
    parser.add_argument(
        '--dataset_root', default='data/cifar-10-batches-bin/', help='dataset root')
    parser.add_argument(
        '--logdir', default='log/resnet_20', help='log directory')
    parser.add_argument('--restore', default='', help='snapshot path')
    parser.add_argument('--validate_rate', default=0.1,
                        type=float, help='validate split rate')

    parser.add_argument('--device', default='cuda:0', help='device')

    parser.add_argument('--model', default='resnet_20',
                        choices=['resnet_20', 'simple_cnn'], help='model name')
    parser.add_argument('--n_epoch', default=70,
                        type=int, help='number of epoch')
    parser.add_argument('--weight_decay', default=0.0001,
                        type=float, help='weight decay rate')
    parser.add_argument('--boundaries', default='30,60',
                        help='learning rate boundaries')
    parser.add_argument(
        '--values', default='1e-1,1e-2,1e-3', help='learning rate values')

    parser.add_argument('--log_every', default=100, type=int,
                        help='display and log frequency')
    parser.add_argument('--seed', default=0, type=float, help='random seed')

    parser.add_argument('--batch_size', default=64,
                        type=int, help='batch size')
    args = parser.parse_args()
    main(args)
