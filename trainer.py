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
import sys
import glob

import traceback
import pdb

class Trainer():
  def __init__(self, logdir='./logs/', savedir='./checkpoints/', _logname='exp', debug=False, benchmark=True, profile_burnin=200, profile_stats=False, num_workers=5, pin_memory=True, **kwargs):
    self.__dict__.update(locals())
    self.__dict__.update(kwargs)
    self._last_log = {}
    if 'postfix' not in kwargs:
      self.postfix = lambda: self._last_log
    
    self.state_dict = {}
    self.state_dict['experiment_uid'] = '{:%m-%d-%H%M%S}'.format(datetime.now())
    self.state_dict['experiment_name'] = '{}-{}'.format(_logname, self.state_dict['experiment_uid'])
    self.state_dict.update({
      'logpath': "{}{}".format(logdir, self.state_dict['experiment_name']),
      'savepath': "{}{}".format(savedir, self.state_dict['experiment_name'])
    })
      

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
        step = trainer.state_dict.get('step', 0)
        init_epoch = trainer.state_dict.get('epoch', 0)
        with tqdm(range(init_epoch, epochs), desc='Epochs', disable=not show_progress or epochs == 1, initial=init_epoch, total=epochs) as epochbar:
          for e in epochbar:
            trainer.state_dict['epoch'] = e

            with tqdm(dataloader, 
                      disable=not show_progress,
                      desc=progress, leave=leave, smoothing=1) as bar:
              
              for i, data in enumerate(bar):
                if trainer.cuda:
                  data = [d.cuda(async=True) for d in data] if len(data) > 0 else d.cuda(async=True)
                after_load = time.clock()

                if before_load > 0:
                  trainer.load_duration = profile_alpha * trainer.load_duration + (1 - profile_alpha) * (after_load - before_load)
                
                with torch.autograd.set_grad_enabled(grad):
                  with torch.autograd.set_detect_anomaly(trainer.debug) as stack:
                    with torch.autograd.profiler.profile(utils.is_profile and step >= trainer.profile_burnin, use_cuda=trainer.cuda) as prof:

                      before_run = time.clock()
                      yield (e, i, step, bar), data
                      after_run = time.clock()

                      trainer.run_duration = profile_alpha * trainer.run_duration + (1 - profile_alpha) * (after_run - before_run)
                      trainer.dataload_fraction = trainer.load_duration / (trainer.run_duration + trainer.load_duration)

                    if utils.is_profile and step >= trainer.profile_burnin:
                      bar.clear()
                      print(prof.key_averages().table('cpu_time_total'))
                      utils.profile.print_stats()
                      exit()

                step += 1

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
    
    if self.profile_stats or utils.is_profile:
      stats = utils.gpustats()
      self._last_log['maxmem'] = '{0:.1%}'.format(stats['maxmemusage'])
      self._last_log['maxutil'] = '{0:.0%}'.format(stats['maxutil'])

      if hasattr(self, 'dataload_fraction'):
        self._last_log['dperf'] = '{0:.1%}'.format(self.dataload_fraction)

    newdict = self.flatten_dict(kwargs)
    logtb = True
    
    if utils.is_profile or self.debug:
      # don't log to tensorboard
      logtb = False

    if not hasattr(self, 'writer') and logtb:
      self.writer = SummaryWriter(self.state_dict['logpath'])

    #self._last_log.update(newdict)
    for label, value in newdict.items():
      if torch.is_tensor(value) and value.numel() == 1:
        self._last_log[label] = value.item()
        self.writer.add_scalar(label, value, t) if logtb else None
      
      elif torch.is_tensor(value) and len(value.size()) == 4:
        image = vutils.make_grid(value, normalize=True, scale_each=True)
        self.writer.add_image(label, image, t) if logtb else None

      elif torch.is_tensor(value) and len(value.size()) == 3:
        self.writer.add_image(label, value, t) if logtb else None

      elif torch.is_tensor(value) and len(value.size()) == 2:
        pass
      #   self.writer.add_embedding(value)

      elif torch.is_tensor(value) and len(value.size()) == 1:
        self.writer.add_histogram(label, value, t) if logtb else None

      elif np.isscalar(value) and not isinstance(value, str):
        self.writer.add_scalar(label, value, t) if logtb else None
        self._last_log[label] = value
      
      elif np.isscalar(value) and isinstance(value, str):
        import re
        value = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -/]*[@-~]').sub('', ('\t' + value).replace('\n', '  \n\t'))
        self.writer.add_text(label, value, t) if logtb else None

      else:
        self._last_log[label] = value
  
  def checkpoint(self, model, optimizer, tag='checkpoint', resume=False, path=None, log=None, **kwargs):
    if not utils.is_profile and not self.debug:
      filename = tag + '.pth.tar'
      path = os.path.join(self.state_dict['savepath'], filename) if path is None else path

      if not os.path.exists(self.state_dict['savepath']):
        os.makedirs(self.state_dict['savepath'])

      sd = {'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'trainer': self.state_dict,
            'extra': kwargs}

      # try to save and handle exceptions by resaving
      # we shouldn't use "finally" just in case torch.save
      # needs to throw an exception
      try:
        torch.save(sd, path)
      except BaseException as e:
        if os.path.exists(path):
          os.remove(path)
        torch.save(sd, path)
        raise type(e)('Interrupt recieved during save. Resaving.') from e

      if log is not None:
        log('Checkpoint at {}: {}'.format(self.state_dict.get('step', None), path))

      return path

  def resume(self, model, optimizer, *args, path=None, uid='', unique=True):
    filename = '.'.join(args) + '.pth.tar'
    if path is None:
      path = os.path.join(self.state_dict['savepath'], filename)
      path = path.replace(self.state_dict['experiment_name'], '*' if uid == '' else uid)
      
    checkpoints = glob.glob(path)
    checkpoints.sort(key=os.path.getmtime)

    if len(checkpoints) == 0:
      raise Exception('No checkpoints found: {}'.format(path))

    if len(checkpoints) == 1 or unique == False:
      path = checkpoints[-1]
      sd = torch.load(path)
      model.load_state_dict(sd['model'])
      optimizer.load_state_dict(sd['optimizer'])

      self.state_dict = sd['trainer']
      return path
      
    else:
      raise Exception('Checkpoint is not unique, but unique=True was specified')
