import torch
from torch.utils.data import DataLoader

class LRStrategy:

    def __init__(self, *args, **kwargs):
        pass

    def get(self, epoch, global_step):
        raise NotImplementedError()

    def set_lr(self, epoch, global_step, optim):
        lr = self.get(epoch, global_step)
        for group in optim.param_groups:
            group['lr'] = lr


class CallbackHandler:

    def __init__(self, trainer, callbacks):
        self.callbacks = callbacks
        self.trainer = trainer

    def on_batch_begin(self):
        for cb in self.callbacks:
            cb.on_batch_begin(self.trainer)

    def on_batch_end(self):
        for cb in self.callbacks:
            cb.on_batch_end(self.trainer)

    def on_epoch_begin(self):
        for cb in self.callbacks:
            cb.on_epoch_begin(self.trainer)

    def on_epoch_end(self):
        for cb in self.callbacks:
            cb.on_epoch_end(self.trainer)


class Trainer:

    def __init__(self, model, step_fn, optim=None, lr_strategy=None):
        self.model = model
        self.optim = optim
        self.step_fn = step_fn
        self.lr_strategy = lr_strategy

        self.env = {}
        self.global_step = 0
        self.epoch = 0

    def _step(self, xs, cb_handler, training=True):
        if training:
            self.optim.zero_grad()

        self.put('xs', xs)
        self.put('global_step', self.global_step)

        cb_handler.on_batch_begin()
        loss = self.step_fn(self.get('xs', xs), self.model,
                            training, trainer=self)

        if training:
            loss.backward()
            self.lr_strategy.set_lr(self.epoch, self.global_step, self.optim)
            self.optim.step()

        cb_handler.on_batch_end()

        if training:
            self.global_step += 1

    def _run_epoch(self, data_iter, n_step=None, callbacks=[], metrics=[], training=True):
        self.put('epoch', self.epoch)
        self.put('metrics', metrics)
        cb_handler = CallbackHandler(self, metrics + callbacks)

        cb_handler.on_epoch_begin()
        for i, xs in enumerate(data_iter):
            self.put('epoch_step', i)
            self._step(xs, cb_handler=cb_handler, training=training)
            if n_step is not None and i == n_step - 1:
                break
        cb_handler.on_epoch_end()

        self.reset()
        if training:
            self.epoch += 1
        return i + 1

    def fit(self, data_iter, n_step=None, callbacks=[], metrics=[]):
        if self.optim is None:
            raise RuntimeError('Otimizer is None.')
        if self.lr_strategy is None:
            raise RuntimeError('LRStrategy is None.')
            
        if not self.model.training:
            self.model.train()
        return self._run_epoch(
            data_iter, n_step=n_step, callbacks=callbacks, metrics=metrics)

    def eval(self, data_iter, n_step=None, callbacks=[], metrics=[]):
        if self.model.training:
            self.model.eval()
        return self._run_epoch(
            data_iter, n_step=n_step, callbacks=callbacks,
            metrics=metrics, training=False)

    def predice(self, *args):
        if self.model.training:
            self.model.eval()
        return self.model(*args)

    def put(self, name, value):
        self.env[name] = value

    def get(self, name, default=None):
        return self.env.get(name, default)

    def get_or_raise(self, name):
        if name in self.env:
            return self.env[name]
        else:
            raise KeyError('"{}" is not in the trainer env.'.format(name))

    def update(self, kdict):
        self.env.update(kdict)

    def reset(self):
        self.env = {}

    def state_dict(self):
        return {
            'model': self.model.state_dict(),
            'optim': self.optim.state_dict() if self.optim else None,
            'epoch': self.epoch,
            'global_step': self.global_step
        }

    def load_state_dict(self, state):
        self.model.load_state_dict(state['model'])
        if self.optim:
            self.optim.load_state_dict(state['optim'])
        self.epoch = state['epoch']
        self.global_step = state['global_step']
