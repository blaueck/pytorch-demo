from argparse import ArgumentParser
import os
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F

import data_utils
import models
from trainer import LRStrategy, Trainer
from callback import ConsoleLogger, EpochCheckpointSaver, MetricCollector
import metrics


class LRManager(LRStrategy):

    def __init__(self, boundaries, values):
        super(LRManager, self).__init__()

        self.boundaries = boundaries
        self.values = values

    def get(self, epoch, global_step):
        for b, v in zip(self.boundaries, self.values):
            if epoch < b:
                return v
        return self.values[-1]


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


def step_fn(xs, model, training, trainer):
    images, labels = xs
    logits = model(images)
    loss = F.cross_entropy(logits, labels)

    trainer.update({
        'loss': loss,
        'logits': logits,
        'labels': labels
    })
    return loss


def main(args):

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    # load data
    data, meta = data_utils.load_data(
        args.dataset_root, args.dataset, is_training=True)
    train_data, val_data = data_utils.split_data(
        data, args.validate_rate, shuffle=True)

    # build train dataloader
    train_dataset = data_utils.ImageDataset(
        *train_data, is_training=True, is_flip=args.dataset not in ['mnist', 'svhn'])
    train_dataloader = data_utils.DeviceDataLoader(
        device, train_dataset, args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # build val dataloader
    val_dataset = data_utils.ImageDataset(*val_data, is_training=False)
    val_dataloader = data_utils.DeviceDataLoader(
        device, val_dataset, args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # remove temp dataset variables to reduce memory usage
    del data, train_data, val_data

    # build model
    if args.model == 'resnet_20':
        model = models.Resnet20
    else:
        model = models.SimpleCNN
    net = model(train_dataset.shape, meta['n_class']).to(device=device)
    net.apply(init_param)

    # build optim
    optim = torch.optim.SGD(make_param_groups(
        net, args.weight_decay), 0.1, momentum=0.9)

    # lr strategy
    lr_boundaries = list(map(int, args.boundaries.split(',')))
    lr_values = list(map(float, args.values.split(',')))
    lr_manager = LRManager(lr_boundaries, lr_values)

    trainer = Trainer(net, step_fn, optim, lr_manager)

    train_collector = MetricCollector(args.log_every)
    train_metrics = [
        metrics.LR(),
        metrics.Mean('loss', args.log_every),
        metrics.Accuracy(args.log_every),
        metrics.Speed(args.batch_size, args.log_every, use_cuda=device.type=='cuda')]
    train_cb = [ConsoleLogger(args.log_every),
                EpochCheckpointSaver(args.logdir),
                train_collector]

    val_collector = MetricCollector()
    val_metrics = [
        metrics.Mean('loss'),
        metrics.Accuracy(),
        metrics.Speed(args.batch_size)]
    val_cb = [ConsoleLogger(), val_collector]

    if args.restore:
        state = torch.load(args.restore)
        trainer.load_state_dict(state)

    for e in range(trainer.epoch, args.n_epoch):
        trainer.fit(train_dataloader, n_step=200, callbacks=train_cb, metrics=train_metrics)
        trainer.eval(val_dataloader, callbacks=val_cb, metrics=val_metrics)
        torch.save({'train': train_collector.state_dict(), 'val': val_collector.state_dict()},
                    os.path.join(args.logdir, 'log_{:d}.pk'.format(e)))


if __name__ == '__main__':
    parser = ArgumentParser(description='pytorch for small image dataset')
    parser.add_argument('--dataset', default='cifar10',
                        help='the training dataset')
    parser.add_argument(
        '--dataset_root', default='../test_adam/data/cifar-10-batches-bin/', help='dataset root')
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
