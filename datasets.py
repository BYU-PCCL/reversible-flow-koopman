from libs.args.args import argchoice

from torchvision import datasets, transforms

class Dataset(datasets.MNIST):
  def __init__(self, root='/tmp/mnist', download=True):
  	super(Dataset, self).__init__(
  		root, 
  		download=download, 
  		transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))