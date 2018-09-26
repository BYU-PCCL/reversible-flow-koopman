from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

class Trainer():
  def __init__(self, model, optimizer, **kwargs):
    self.__dict__.update(locals())
    self.__dict__.update(kwargs)
    self._last_log = {}
    self.s = 0
    if 'postfix' not in kwargs:
      self.postfix = lambda: self._last_log
  
  def __call__(self, dataset, epochs=1, train=False, progress=None, leave=True, **kwargs):
    trainer = self
    dataloader = DataLoader(dataset, **kwargs)
    
    class Gen():
      def __len__(self):
        return epochs * len(dataloader)
      
      def __iter__(self):
        show_progress = progress is not None
        s = 0
        for e in tqdm(range(epochs),
                      disable=not show_progress or epochs == 1):
          
          with tqdm(dataloader, 
                    disable=not show_progress,
                    desc=progress, leave=leave, miniters=1000) as bar:
            
            for i, data in enumerate(bar):
              
              yield (e, i, s), data

              s += 1

              # only call trainer.postfix if the bar was updated
              # this means the display is always 1 behind 
              if progress and bar.last_print_n == i:
                bar.set_postfix(**trainer.postfix(), refresh=False)

    return Gen()
        
  @staticmethod
  def flatten_dict(dd, separator='.', prefix=''):
    # https://stackoverflow.com/questions/6027558/flatten-nested-python-dictionaries-compressing-keys
    return { prefix + separator + k if prefix else k : v
             for kk, vv in dd.items()
             for k, v in Trainer.flatten_dict(vv, separator, kk).items()
             } if isinstance(dd, dict) else { prefix : dd }
  
  def log(self, t, **kwargs):
    self._last_log = self.flatten_dict(kwargs)
    for label, value in self._last_log.items():
      if torch.is_tensor(value) and value.numel() == 1:
        self._last_log[label] = value.item()
  
  def checkpoint(self):
    pass
  
  def resume(self):
    pass