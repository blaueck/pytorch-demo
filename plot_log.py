import sys
from pathlib import Path
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import torch


def parse_bool(value):
    if isinstance(value, bool):
        return value
    elif isinstance(value, str):
        value = value.lower()
        if value in ['true', 'ok', '1']:
            return True
    return False


def plot_dir(logdir, prefix):
    logfiles = list(logdir.glob('log_*'))

    if len(logfiles) == 0:
        return

    # collect log
    train_log, val_log = {}, {}
    for log_f in logfiles:
        log = torch.load(str(log_f))
        train_log.update(log['train'])
        val_log.update(log['val'])

    # prepair logs for plot
    train_step, train_loss, train_acc = [], [], []
    for k in sorted(train_log.keys()):
        train_step.append(k)
        data = train_log[k]
        train_loss.append(data['loss'])
        train_acc.append(data['acc'])

    val_step, val_loss, val_acc = [], [], []
    for k in sorted(val_log.keys()):
        val_step.append(k)
        data = val_log[k]
        val_loss.append(data['loss'])
        val_acc.append(data['acc'])
    
    # plot logs
    plt.subplot(121)
    plt.title('Loss')
    plt.plot(train_step, train_loss, '-', label=prefix+'train')
    plt.plot(val_step, val_loss, '-', label=prefix+'val')
    plt.xlabel('step')
    plt.legend()

    plt.subplot(122)
    plt.title('Accuracy')
    plt.plot(train_step, train_acc, '-', label=prefix+'train')
    plt.plot(val_step, val_acc, '-', label=prefix+'val')
    plt.xlabel('step')
    plt.legend()


def main(args):
    logdir = Path(args.logdir)
    
    dirs = []
    if args.multi_dir:
        dirs = [d for d in logdir.iterdir() if d.is_dir()]
    else:
        dirs = [logdir]

    for d in dirs:
        subfix = d.name + '-' if args.multi_dir else ''
        plot_dir(d, subfix)

    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser(description='plot training log')
    parser.add_argument('logdir', help='log directory')
    parser.add_argument('-m', '--multi_dir', type=parse_bool, default=False,
                        help='whether logdir contain multi sub-directory that contain logs')
    args = parser.parse_args()
    sys.exit(main(args))
