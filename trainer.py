from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import contextlib
import time
import utils
import signal

class Trainer():
  def __init__(self, debug=False, profile=False, benchmark=True, profile_delay=200, profile_stats=False, num_workers=0, pin_memory=True, **kwargs):
    self.__dict__.update(locals())
    self.__dict__.update(kwargs)
    self._last_log = {}
    if 'postfix' not in kwargs:
      self.postfix = lambda: self._last_log

    torch.backends.cudnn.benchmark=benchmark

  def cuda(self):
    self.cuda = True
  
  def __call__(self, dataset, epochs=1, train=False, progress=None, leave=True, profile_alpha=.90, **kwargs):
    trainer = self
    dkwargs = {'num_workers': self.num_workers, 
               'pin_memory':self.pin_memory,

               # Hide KeyboardInterrupt on the workers
               'worker_init_fn': lambda x: signal.signal(signal.SIGINT, lambda signum, frame: None)}
    dkwargs.update(kwargs)

    dataloader = DataLoader(dataset, **dkwargs)
    
    class Gen():
      def __len__(self):
        return epochs * len(dataloader)
      
      def __iter__(self):
        show_progress = progress is not None
        s = 0
        trainer.load_duration = 0
        trainer.run_duration = 0
        trainer.dataload_fraction = 0
        for e in tqdm(range(epochs),
                      disable=not show_progress or epochs == 1):
          
          with tqdm(dataloader, 
                    disable=not show_progress,
                    desc=progress, leave=leave) as bar:
            
            for i, data in enumerate(bar):
              if trainer.cuda:
                data = [d.cuda(async=True) for d in data] if len(data) > 0 else d.cuda(async=True)
              after_load = time.clock()

              if s > 0:
                trainer.load_duration = profile_alpha * trainer.load_duration + (1 - profile_alpha) * (after_load - before_load)
                          
              with torch.autograd.set_detect_anomaly(trainer.debug) as stack:
                with torch.autograd.profiler.profile(trainer.profile and s >= trainer.profile_delay, use_cuda=trainer.cuda) as prof:

                  before_run = time.clock()
                  yield (e, i, s), data
                  after_run = time.clock()

                  trainer.run_duration = profile_alpha * trainer.run_duration + (1 - profile_alpha) * (after_run - before_run)
                  trainer.dataload_fraction = trainer.load_duration / (trainer.run_duration + trainer.load_duration)

                if trainer.profile and s >= trainer.profile_delay:
                  bar.clear()

                  print(prof.key_averages().table('cpu_time_total'))

                  exit()

              s += 1

              # only call trainer.postfix if the bar was updated
              # this means the display is always 1 behind 
              if progress and bar.last_print_n == i:
                bar.set_postfix(**trainer.postfix(), refresh=False)

              before_load = time.clock()

    return Gen()
        
  @staticmethod
  def flatten_dict(dd, separator='.', prefix=''):
    # https://stackoverflow.com/questions/6027558/flatten-nested-python-dictionaries-compressing-keys
    return { prefix + separator + k if prefix else k : v
             for kk, vv in dd.items()
             for k, v in Trainer.flatten_dict(vv, separator, kk).items()
             } if isinstance(dd, dict) else { prefix : dd }
  
  def log(self, t, **kwargs):
    self._last_log.update(self.flatten_dict(kwargs))
    for label, value in self._last_log.items():
      if torch.is_tensor(value) and value.numel() == 1:
        self._last_log[label] = value.item()

    if self.profile_stats:
      stats = utils.gpustats()
      self._last_log['maxgpumem'] = '{0:.1%}'.format(stats['maxmemusage'])
      self._last_log['maxgpuutil'] = '{0:.0%}'.format(stats['maxutil'])
      self._last_log['dataperf'] = '{0:.1%}'.format(self.dataload_fraction)
  
  def checkpoint(self):
    pass
  
  def resume(self):
    pass