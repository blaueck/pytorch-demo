import os
import torch

class Callback:

    def __init__(self, *args, **kwargs):
        pass

    def on_batch_begin(self, trainer):
        pass

    def on_batch_end(self, trainer):
        pass

    def on_epoch_begin(self, trainer):
        pass

    def on_epoch_end(self, trainer):
        pass


class ConsoleLogger(Callback):

    def __init__(self, interval=None):
        super(ConsoleLogger, self).__init__()
        self.interval = interval
    
    def on_batch_end(self, trainer):
        if self.interval is None:
            return 

        epoch_step = trainer.get_or_raise('epoch_step')
        if epoch_step % self.interval == 0:
            outstr = 'step: {:d}, '.format(epoch_step)
            print(self._get_metrics_str(trainer, outstr))
    
    def on_epoch_begin(self, trainer):
        if self.interval is None:
            return 
        print('---------Epoch: {:d}----------'
              .format(trainer.get_or_raise('epoch')))

    def on_epoch_end(self, trainer):
        if self.interval is None:
            print(self._get_metrics_str(trainer, ''))

    def _get_metrics_str(self, trainer, prefix):
        metrics = trainer.get_or_raise('metrics')
        values = {}
        formats = {}
        for m in metrics:
            values.update(m.get())
            formats.update(m.format_str())
        values_strs = []
        for k in values:
            values_strs.append(('{}: '+formats[k]).format(k, values[k]))
        return prefix + ', '.join(values_strs)


class EpochCheckpointSaver(Callback):

    def __init__(self, logdir, cp_fmt='checkpoint_{:d}.pk'):
        super(EpochCheckpointSaver, self).__init__()
        self.logdir = logdir
        self.cp_fmt = cp_fmt

        if not os.path.exists(logdir):
            os.makedirs(logdir)
    
    def on_epoch_end(self, trainer):
        epoch = trainer.get_or_raise('epoch')
        cp_path = os.path.join(self.logdir, self.cp_fmt.format(epoch))
        torch.save(trainer.state_dict(), cp_path)
    

class MetricCollector(Callback):

    def __init__(self, interval=None):
        super(MetricCollector, self).__init__()
        self.interval = interval
        self.state = {}
    
    def on_batch_end(self, trainer):
        if self.interval is None:
            return 

        epoch_step = trainer.get_or_raise('epoch_step')
        if epoch_step % self.interval == 0:
            self._collect(trainer)

    def on_epoch_end(self, trainer):
        if self.interval is None:
            self._collect(trainer)

    def _collect(self, trainer):
        metrics = trainer.get_or_raise('metrics')
        global_step = trainer.get_or_raise('global_step')
        values = {}
        for m in metrics:
            values.update(m.get())
        self.state[global_step] = values
    
    def state_dict(self):
        return self.state