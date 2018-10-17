from libs.args.args import argchoice
import numpy as np
import torch
from torchvision import datasets, transforms

class Dataset(datasets.MNIST):
  def __init__(self, root='/tmp/mnist', train=True, download=True, overfit=False):
  	self.overfit = overfit
  	super(Dataset, self).__init__(
  		root, 
  		download=download, 
      train=train,
  		transform=transforms.Compose([
                           transforms.Resize((32,32)),
                           transforms.ToTensor(),
                           lambda x: torch.cat([x, x[0:1]], dim=0),
                           #lambda x: x + torch.normal(mean=torch.zeros_like(x), std=torch.ones_like(x) * (1. / 256.))
                           lambda x: x + torch.zeros_like(x).uniform_(0., 1./ 255.),
                           #transforms.Normalize((0.1715,), (1.0,))
                       ]))

  def __len__(self):
  	if self.overfit:
  		return 32
  	else:
  		return super(Dataset, self).__len__()