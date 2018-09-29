from libs.args.args import argchoice

from torchvision import datasets, transforms

class Dataset(datasets.MNIST):
  def __init__(self, root='/tmp/mnist', download=True, overfit=False):
  	self.overfit = overfit
  	super(Dataset, self).__init__(
  		root, 
  		download=download, 
  		transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
  def __len__(self):
  	if self.overfit:
  		return 1
  	else:
  		return super(Dataset, self).__len__()