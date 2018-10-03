from libs.args.args import argchoice
import numpy as np
import torch
from torchvision import datasets, transforms

class Dataset(datasets.MNIST):
  def __init__(self, root='/tmp/mnist', download=True, overfit=False):
  	self.overfit = overfit
  	super(Dataset, self).__init__(
  		root, 
  		download=download, 
  		transform=transforms.Compose([
                           transforms.ToTensor(),
                           lambda x: torch.cat([x, x[0:1]], dim=0),
                           lambda x: x + torch.zeros_like(x).uniform_(0., 1./ 255.),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
  def __len__(self):
  	if self.overfit:
  		return 1
  	else:
  		return super(Dataset, self).__len__()