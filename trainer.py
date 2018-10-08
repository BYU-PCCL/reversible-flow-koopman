from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import contextlib
import time
from datetime import datetime
import utils
import signal
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import numpy as np
import builtins
import os
import glob

import pdb

class Trainer():
  def __init__(self, logdir='./logs/', savedir='./checkpoints/', _logname='exp', debug=False, benchmark=True, profile_burnin=200, profile_stats=False, num_workers=5, pin_memory=True, **kwargs):
    self.__dict__.update(locals())
    self.__dict__.update(kwargs)
    self._last_log = {}
    if 'postfix' not in kwargs:
      self.postfix = lambda: self._last_log
    
    self.experiment_uid = '{:%m-%d-%H%M%S}'.format(datetime.now())
    self.experiment_name = '{}-{}'.format(_logname, self.experiment_uid)
    self.logpath = "{}{}".format(logdir, self.experiment_name)
    self.savepath = "{}{}".format(savedir, self.experiment_name)
    self.writer = SummaryWriter(self.logpath)
    self.state_dict = {}

    torch.backends.cudnn.benchmark=benchmark

  def cuda(self):
    self.cuda = True
  
  def __call__(self, dataset, epochs=1, train=False, progress=None, leave=True, profile_alpha=.90, grad=True, **kwargs):
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
        trainer.load_duration = 0
        trainer.run_duration = 0
        trainer.dataload_fraction = 0
        before_load = -1
        with tqdm(range(epochs), desc='Epochs', disable=not show_progress or epochs == 1) as epochbar:
          for e in epochbar:
            trainer.state_dict['epoch'] = e
            with tqdm(dataloader, 
                      disable=not show_progress,
                      desc=progress, leave=leave) as bar:
              
              for i, data in enumerate(bar):
                if trainer.cuda:
                  data = [d.cuda(async=True) for d in data] if len(data) > 0 else d.cuda(async=True)
                after_load = time.clock()

                if before_load > 0:
                  trainer.load_duration = profile_alpha * trainer.load_duration + (1 - profile_alpha) * (after_load - before_load)
                
                with torch.autograd.set_grad_enabled(grad):
                  with torch.autograd.set_detect_anomaly(trainer.debug) as stack:
                    with torch.autograd.profiler.profile(utils.is_profile and trainer.state_dict.get('step', 0) >= trainer.profile_burnin, use_cuda=trainer.cuda) as prof:

                      before_run = time.clock()
                      yield (e, i, trainer.state_dict.get('step', 0)), data
                      after_run = time.clock()

                      trainer.run_duration = profile_alpha * trainer.run_duration + (1 - profile_alpha) * (after_run - before_run)
                      trainer.dataload_fraction = trainer.load_duration / (trainer.run_duration + trainer.load_duration)

                    if utils.is_profile and trainer.state_dict.get('step', 0) >= trainer.profile_burnin:
                      bar.clear()
                      print(prof.key_averages().table('cpu_time_total'))
                      utils.profile.print_stats()
                      exit()

                trainer.state_dict['step'] = trainer.state_dict.get('step', 0) + 1

                # only call trainer.postfix if the bar was updated
                # this means the display is always 1 behind 
                if progress and bar.last_print_n == i:
                  bar.set_postfix(**trainer.postfix(), refresh=False)

                before_load = time.clock()

            epochbar.set_postfix(**trainer.postfix(), refresh=True)

    return Gen()
        
  @staticmethod
  def flatten_dict(dd, separator='.', prefix=''):
    # https://stackoverflow.com/questions/6027558/flatten-nested-python-dictionaries-compressing-keys
    return { prefix + separator + k if prefix else k : v
             for kk, vv in dd.items()
             for k, v in Trainer.flatten_dict(vv, separator, kk).items()
             } if isinstance(dd, dict) else { prefix : dd }
  
  def log(self, t, **kwargs):
    newdict = self.flatten_dict(kwargs)
    #self._last_log.update(newdict)
    for label, value in newdict.items():
      if torch.is_tensor(value) and value.numel() == 1:
        self._last_log[label] = value.item()
        self.writer.add_scalar(label, value.item(), t)
      
      elif torch.is_tensor(value) and len(value.size()) == 4:
        image = vutils.make_grid(value, normalize=True, scale_each=True)
        self.writer.add_image(label, image, t)

      elif torch.is_tensor(value) and len(value.size()) == 3:
        self.writer.add_image(label, value, t)

      elif torch.is_tensor(value) and len(value.size()) == 2:
        pass
      #   self.writer.add_embedding(value)

      elif torch.is_tensor(value) and len(value.size()) == 1:
        self.writer.add_histogram(label, value, t)

      elif np.isscalar(value) and not isinstance(value, str):
        self.writer.add_scalar(label, value, t)
        self._last_log[label] = value
      
      elif np.isscalar(value) and isinstance(value, str):
        import re
        value = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -/]*[@-~]').sub('', ('\t' + value).replace('\n', '  \n\t'))
        self.writer.add_text(label, value, t)

      else:
        self._last_log[label] = value

    if self.profile_stats:
      stats = utils.gpustats()
      self._last_log['maxmem'] = '{0:.1%}'.format(stats['maxmemusage'])
      self._last_log['maxutil'] = '{0:.0%}'.format(stats['maxutil'])

      if hasattr(self, 'dataload_fraction'):
        self._last_log['dperf'] = '{0:.1%}'.format(self.dataload_fraction)
  
  def checkpoint(self, model, *args, resume=False, path=None, **kwargs):
    if len(args) == 0 and len(kwargs) == 0:
      args = ['checkpoint']

    filename = '.'.join(args) + '.pth.tar'
    path = os.path.join(self.savepath, filename) if path is None else path

    if not os.path.exists(self.savepath):
      os.makedirs(self.savepath)
    
    # if folder has too many files (or is too large)
    # delete oldest
    sd = {'checkpoint': model.state_dict(),
          'trainer': self.state_dict,
          'extra': kwargs}

    torch.save(sd, path)

    return True

  def resume(self, model, *args, path=None, uid='', unique=True):
    filename = '.'.join(args) + '.pth.tar'
    if path is None:
      path = os.path.join(self.savepath, filename)
      path = path.replace(self.experiment_uid if uid == '' else uid, '*')

    checkpoints = glob.glob(path)
    checkpoints.sort(key=os.path.getmtime)

    if len(checkpoints) == 0:
      return False

    if len(checkpoints) == 1 or unique == False:
      path = checkpoints[-1]
      sd = torch.load(path)
      model.load_state_dict(sd['checkpoint'])
      self.state_dict = sd['trainer']
      return True
      
    else:
      raise IndexError('Checkpoint is not unique, but unique=True was specified')
