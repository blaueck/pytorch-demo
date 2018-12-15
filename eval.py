from argparse import ArgumentParser
import os
from pathlib import Path

import torch

import data_utils
import models
import metrics
from train import step_fn
from trainer import Trainer
from callback import ConsoleLogger


def main(args):

    device = torch.device(args.device)

    # load data
    data, meta = data_utils.load_data(
        args.dataset_root, args.dataset, is_training=False)

    # build val dataloader
    dataset = data_utils.ImageDataset(*data, is_training=False)
    dataloader = data_utils.DeviceDataLoader(
        device, dataset, args.batch_size, shuffle=False, num_workers=2)

    # remove temp dataset variables to reduce memory usage
    del data

    # build model
    if args.model == 'resnet_20':
        model = models.Resnet20
    else:
        model = models.SimpleCNN
    net = model(dataset.shape, meta['n_class']).to(device=device)

    trainer = Trainer(net, step_fn)

    state = torch.load(args.cpt)
    trainer.load_state_dict(state)

    eval_metrics = [
        metrics.Mean('loss'),
        metrics.Accuracy()]

    eval_cb = [ConsoleLogger()]
    trainer.eval(dataloader, callbacks=eval_cb, metrics=eval_metrics)


if __name__ == '__main__':
    parser = ArgumentParser(description='pytorch for small image dataset')
    parser.add_argument('cpt', default='', help='checkpoint path')
    parser.add_argument('--dataset', default='cifar10',
                        help='the eval dataset')
    parser.add_argument(
        '--dataset_root', default='data/cifar-10-batches-bin/', help='dataset root')
    
    parser.add_argument('--model', default='resnet_20',
                        choices=['resnet_20', 'simple_cnn'], help='model name')

    parser.add_argument('--device', default='cuda:0', help='device')

    parser.add_argument('--batch_size', default=64,
                        type=int, help='batch size')
    args = parser.parse_args()
    main(args)
