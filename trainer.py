from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import contextlib
import time
from datetime import datetime
import utils
import signal
from tensorboardX import SummaryWriter, utils as tbxutils
import torchvision.utils as vutils
import numpy as np
import builtins
import os
import sys
import glob
import matplotlib
from matplotlib import pyplot as plt

import traceback
import pdb

class Trainer():
  def __init__(self, logdir='./logs/', savedir='./checkpoints/', _logname='exp', debug=False, benchmark=True, profile_burnin=100, profile_stats=False, num_workers=5, pin_memory=True, **kwargs):
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

    self._cuda = False

    torch.backends.cudnn.benchmark=benchmark

  def _processpostfix(self):
    d = self.postfix()
    d = {k.split(':')[0]:d[k] for k in d if k.split('.')[-1][0] != ':'}
    d = {k:(d[k].item() if torch.is_tensor(d[k]) else d[k]) for k in d}
    return d

  def cuda(self):
    self._cuda = True

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
                      desc=progress, leave=leave, smoothing=.9) as bar:
              
              for i, data in enumerate(bar):
                if trainer._cuda:
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
                if show_progress and bar.last_print_n == i:
                  bar.set_postfix(**trainer._processpostfix(), refresh=True)

                before_load = time.clock()

            epochbar.set_postfix(**trainer._processpostfix(), refresh=True)

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
      self._last_log['maxmem:'] = '{0:.1%}'.format(stats['maxmemusage'])
      self._last_log['maxutil:'] = '{0:.0%}'.format(stats['maxutil'])

      if hasattr(self, 'dataload_fraction'):
        self._last_log['dperf:'] = '{0:.1%}'.format(self.dataload_fraction)

    newlog = self.flatten_dict(kwargs)
    self._last_log.update(newlog)

    if not hasattr(self, 'writer') and not self.debug:
      self.writer = SummaryWriter(self.state_dict['logpath'])

    # only write log entries for new data
    for label_stub, value in newlog.items():
      label = label_stub.split(':')[-1]
      dotindex = label_stub.rfind('.')
      if dotindex > -1 and label != label_stub and len(label) > 0:
        label = label_stub[:dotindex + 1] + label

      if len(label) > 0 and self.writer:
        if torch.is_tensor(value) and value.numel() == 1:
          self.writer.add_scalar(label, value, t)
        
        elif torch.is_tensor(value) and len(value.size()) == 4:
          value = value[:, :3] if value.size(1) > 3 else value[:, 0:1]
          image = vutils.make_grid(value, normalize=True, scale_each=True)
          # image = image.permute(1, 2, 0)

          # fig = plt.figure()
          # plot = fig.add_subplot(111)
          # plt.imshow(image.detach().cpu())
          # cb = plt.colorbar()
          # plt.axis('off')
          # plt.margins(0)
          # plt.tight_layout()
          # plt.subplots_adjust(left=0, right=1, top=0.99, bottom=0.01)
          # fig.canvas.draw()

          self.writer.add_image(label, image, t) # tbxutils.figure_to_image(fig)
          #plt.close(fig)

        elif torch.is_tensor(value) and len(value.size()) == 3:
          value = value[:3] if value.size(0) > 3 else value[0:1]
          image = value - value.min()
          image /= image.max()
          # fig = plt.figure()
          # plot = fig.add_subplot(111)

          # img = value.permute(1, 2, 0).detach().cpu().numpy()
          # if img.shape[2] == 1:
          #   plt.imshow(img.mean(axis=2), interpolation='nearest')
          # else:
          #   plt.imshow(img, interpolation='nearest')
          
          # cb = plt.colorbar()
          # cb.locator = matplotlib.ticker.MaxNLocator(nbins=5)
          # #cb.ax.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
          # cb.update_ticks()

          # plt.axis('off')
          # plt.margins(0)
          # plt.tight_layout()
          # plt.subplots_adjust(left=0, right=1, top=.99, bottom=.01)
          # fig.canvas.draw()

          self.writer.add_image(label, image, t) # tbxutils.figure_to_image(fig)
          # plt.close(fig)

        elif torch.is_tensor(value) and len(value.size()) == 2:
          pass
        #   self.writer.add_embedding(value)

        elif torch.is_tensor(value) and len(value.size()) == 1:
          self.writer.add_histogram(label, value, t)

        elif np.isscalar(value) and not isinstance(value, str):
          self.writer.add_scalar(label, value, t)
        
        elif np.isscalar(value) and isinstance(value, str):
          import re
          value = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -/]*[@-~]').sub('', ('\t' + value).replace('\n', '  \n\t'))
          self.writer.add_text(label, value, t)
    
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
