from argparse import ArgumentParser
import os
from pathlib import Path

import torch
import torch.utils.data

import data_utils
import models
from metrics import MeanValue, Accuracy


def main(args):

    # load data
    data, meta = data_utils.load_data(
        args.dataset_root, args.dataset, is_training=False)

    # build val dataloader
    dataset = data_utils.ImageDataset(*data, is_training=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # remove temp dataset variables to reduce memory usage
    del data

    device = torch.device(args.device)

    # build model
    if args.model == 'resnet_20':
        model = models.Resnet20
    else:
        model = models.SimpleCNN
    net = model(dataset.shape, meta['n_class']).to(device=device)

    criterion = torch.nn.CrossEntropyLoss()

    state = torch.load(args.cpt)
    net.load_state_dict(state['net'])

    net.eval()
    mean_loss, acc = MeanValue(), Accuracy()
    for x, y in dataloader:

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

    print('loss: {:.4f}, acc: {:.2%}'.format(
        mean_loss.get(), acc.get()))


if __name__ == '__main__':
    parser = ArgumentParser(description='pytorch for small image dataset')
    parser.add_argument('cpt', default='', help='checkpoint path')
    parser.add_argument('--dataset', default='cifar10',
                        help='the eval dataset')
    parser.add_argument(
        '--dataset_root', default='../test_adam/data/cifar-10-batches-bin/', help='dataset root')
    
    parser.add_argument('--model', default='resnet_20',
                        choices=['resnet_20', 'simple_cnn'], help='model name')

    parser.add_argument('--device', default='cuda:0', help='device')

    parser.add_argument('--batch_size', default=64,
                        type=int, help='batch size')
    args = parser.parse_args()
    main(args)
