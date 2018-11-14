from libs.args.args import argchoice
import numpy as np
import torch
from torchvision import datasets, transforms

class MnistDataset(datasets.MNIST):
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


class LTI2DSequence(torch.utils.data.Dataset):
  def __init__(self, train=True, size=10000, sequence_length=4, channels=4, state_dim=64 + 15, observation_dim=64):
    self.__dict__.update(locals())
    rand_state = np.random.get_state()
    np.random.seed(42)
    self.state_dim = state_dim * channels
    self.observation_dim = observation_dim * channels
    self.A = np.random.randn(self.state_dim, self.state_dim)
    self.C = np.zeros([self.observation_dim, self.state_dim])
    self.C[:self.observation_dim, :self.observation_dim] = np.eye(self.observation_dim)
    self.x0 = np.random.randn(size, self.state_dim)
    self.state_noise = np.random.randn(self.sequence_length, self.state_dim)
    np.random.set_state(rand_state)

  def __getitem__(self, idx):
    obs = np.empty([self.sequence_length, self.observation_dim])
    x = self.x0[idx]
    for t in range(self.sequence_length):
      x = self.A.dot(self.x0[idx])
      y = self.C.dot(x)
      obs[t] = y

    k = int(np.sqrt(self.observation_dim / self.channels))
    obs = obs.reshape(self.sequence_length, self.channels, k, k).astype(np.float32) / 100

    return torch.from_numpy(obs), torch.from_numpy(obs[-1])

  def __len__(self):
    return self.x0.shape[0]