import time

import numpy as np


class MeanValue:

    def __init__(self):
        self.value = 0
        self.counter = 0
    
    def add(self, value):
        self.value += value
        self.counter += 1
    
    def get(self):
        return self.value / self.counter
    
    def reset(self):
        self.value = 0
        self.counter = 0


class Accuracy:

    def __init__(self):
        self.n_correct = 0.
        self.n_sample = 0.
    
    def add(self, predicts, labels):
        self.n_sample += len(labels)
        self.n_correct += np.sum(predicts == labels)
    
    def get(self):
        return self.n_correct / self.n_sample
    
    def reset(self):
        self.n_correct = 0.
        self.n_sample = 0.


class TimeMeter:

    def __init__(self):
        self.start_time, self.duration, self.counter = 0., 0., 0.

    def add_counter(self):
        self.counter += 1

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        self.duration += time.perf_counter() - self.start_time

    def get(self):
        return self.duration / self.counter

    def reset(self):
        self.start_time, self.duration, self.counter = 0., 0., 0.