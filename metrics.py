import time

import numpy as np
import torch
from callback import Callback


class Metric(Callback):

    def get(self):
        raise NotImplementedError()
    
    def format_str(self):
        raise NotImplementedError()


class LR(Metric):

    def __init__(self):
        super(LR, self).__init__()
        self.lr = 0.
        self.name = 'lr'

    def on_batch_end(self, trainer):
        epoch = trainer.get_or_raise('epoch')
        global_step = trainer.get_or_raise('global_step')
        self.lr = trainer.lr_strategy.get(epoch, global_step)
    
    def get(self):
        return {self.name: self.lr}
    
    def format_str(self):
        return {self.name: '{:g}'}


class Mean(Metric):

    def __init__(self, name, interval=None, format='{:.4f}'):
        super(Mean, self).__init__()
        self.interval = interval
        self.name = name
        self.format = format
        self.sum = 0
        self.count = 0
    
    def on_batch_end(self, trainer):
        global_step = trainer.get_or_raise('global_step')
        if self.interval and global_step % (self.interval+1) == 0:
            self.reset()
        value = trainer.get_or_raise(self.name)
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        self.sum += value
        self.count += 1
    
    def get(self):
        value = 0
        if self.count > 0:
            value = self.sum / self.count
        return {self.name: value}
    
    def format_str(self):
        return {self.name: self.format}
    
    def reset(self):
        self.sum = 0
        self.count = 0


class Accuracy(Metric):

    def __init__(self, interval=None):
        super(Accuracy, self).__init__()
        self.interval = interval
        self.n_correct = 0
        self.count = 0
        self.name = 'accuracy'
    
    def on_batch_end(self, trainer):
        # reset
        global_step = trainer.get_or_raise('global_step')
        if self.interval and global_step % (self.interval+1) == 0:
            self.reset()

        logits = trainer.get_or_raise('logits')
        labels = trainer.get_or_raise('labels')

        logits = logits.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        predicts = np.argmax(logits, axis=1)

        self.n_correct += np.sum(predicts == labels)
        self.count += len(labels)
    
    def get(self):
        value = 0
        if self.count > 0:
            value = self.n_correct / self.count
        return {self.name: value}
    
    def format_str(self):
        return {self.name: '{:.2%}'}
    
    def reset(self):
        self.n_correct = 0
        self.count = 0


class Speed(Metric):

    def __init__(self, batch_size, interval=None, use_cuda=False):
        super(Speed, self).__init__()
        self.interval = interval
        self.start_time, self.count = 0., 0.
        self.name = 'speed'
        self.use_cuda = use_cuda
        self.batch_size = batch_size
    
    def on_epoch_begin(self, trainer):
        self.reset()
        self.start_time = time.time()
    
    def on_batch_begin(self, trainer):
        # reset
        epoch_step = trainer.get_or_raise('epoch_step')
        if self.interval and epoch_step % (self.interval+1) == 0:
            self.reset()
            self.start_time = time.time()

    def on_batch_end(self, trainer):
        self.count += 1
    
    def get(self):
        spd = 0
        if self.count > 0:
            if self.use_cuda:
                torch.cuda.synchronize()
            value = (time.time() - self.start_time) / self.count
            spd = self.batch_size / value
        return {self.name: spd}
    
    def format_str(self):
        return {self.name: '{:.2f} i/s'}
    
    def reset(self):
        self.start_time, self.count = 0., 0.