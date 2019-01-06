from libs.args.args import argchoice
import numpy as np
import torch
from torchvision import datasets, transforms
import os

import libs.args.args as args

# class MnistDataset(datasets.MNIST):
#   def __init__(self, root='/tmp/mnist', train=True, download=True, overfit=False):
#   	self.overfit = overfit
#   	super(MnistDataset, self).__init__(
#   		root, 
#   		download=download, 
#       train=train,
#   		transform=transforms.Compose([
#                            transforms.Resize((32,32)),
#                            transforms.ToTensor(),
#                            lambda x: torch.cat([x, x[0:1]], dim=0),
#                            #lambda x: x + torch.normal(mean=torch.zeros_like(x), std=torch.ones_like(x) * (1. / 256.))
#                            lambda x: x + torch.zeros_like(x).uniform_(0., 1./ 255.),
#                            #transforms.Normalize((0.1715,), (1.0,))
#                        ]))

#   def __len__(self):
#   	if self.overfit:
#   		return 32
#   	else:
#   		return super(Dataset, self).__len__()

class RotatingCube(torch.utils.data.Dataset):
    def __init__(self, root="/mnt/pccfs/not_backed_up/data/cube_data/spherecube_const_pitch_yaw_16", sequence_length=10, num_channels=3, 
                  transformation=transforms.Compose([transforms.ToTensor()]), 
                  projection_dimensions=2 * 8 * 8, use_pca=False, overfit=False, repeats=10000, train=False):
      super(RotatingCube, self).__init__()
      
      self.repeats = repeats
      self.overfit = overfit
      if overfit:
        self.size = args.reader().batch_size
      
      # Set up image loader
      self.dataset_folder = datasets.ImageFolder(os.path.join(root) ,transform = transformation)        
      self.total_num_frames = len(self.dataset_folder)
      self.sequence_length = sequence_length
      self.use_pca = use_pca

      c, h, w = self.dataset_folder[0][0].size()
      
      # Create data, which is a tensor containing each image sequentially. Length is the number of files in the folder
      # c + (c % 2)
      self.data = torch.zeros([self.total_num_frames, c, h, w]).float()
      for i in range(self.total_num_frames):
          self.data[i] = self.dataset_folder[i][0]
      if use_pca:
        two_d = self.data.view(self.total_num_frames, -1)

        assert self.total_num_frames > projection_dimensions, f"Number of projection dimensions ({projection_dimensions}) is greater than number of samples ({self.total_num_frames}), using number of samples"
        assert projection_dimensions % 2 == 0, 'projection_dimensions must be even'

        # A PCA Method could be included here if desired. 
        # self.data = PCA(self.data)
        # preprocess the data
        X_mean = torch.mean(two_d,0)
        two_d = two_d - X_mean.expand_as(two_d)
        self.U, self.S, self.V = torch.svd(torch.t(two_d), some=True)
        self.data = torch.mm(two_d, self.U[:,:projection_dimensions]) # k is num dimensions
        self.projection_dimensions = projection_dimensions

      #self.data += (torch.rand_like(self.data) - .5) * 1 / 256

      self.centering_const = self.data.view(self.total_num_frames, -1).mean()
      self.normalizing_const = (self.data - self.centering_const).abs().mean()

      #self.data = torch.randn_like(self.data)

    def denormalize(self, x):
      return x * self.normalizing_const + self.centering_const

    def __getitem__(self,index):
      index = index % ((self.total_num_frames - self.sequence_length + 1) if not self.overfit else self.size)
      raw = self.data[index: index + self.sequence_length]
      raw += (torch.rand_like(raw) - .5) * 1 / 256.
      raw = (raw - self.centering_const) / self.normalizing_const

      if self.use_pca:
        k = int(np.sqrt(raw.size(1) / 2))
        return raw.view(self.sequence_length, 2, k, -1), raw[0,0]
      else:
        return raw, raw[0,0]

    def __len__(self):
      if self.overfit:
        # Total number of sequences = Total Frames - Sequence Length + 1
        return self.size * self.repeats

      return (self.total_num_frames - self.sequence_length + 1) * self.repeats

    def __repr__(self):
      summary =   '                   Name: {}\n'.format(self.__class__.__name__)
      summary += '                    Size: {}\n'.format(self.__getitem__(0)[0].shape)
      summary += '               Min Value: {:.2f}\n'.format(self.__getitem__(0)[0].min().item())
      summary += '               Max Value: {:.2f}\n'.format(self.__getitem__(0)[0].max().item())

      if self.use_pca:
        summary += '    Projection Dimension: {}\n'.format(self.projection_dimensions)
        summary += '                 Max SVs: {:.2f}\n'.format(self.S[:self.projection_dimensions].max().item())
        summary += '                 Min SVs: {:.2f}\n'.format(self.S[:self.projection_dimensions].min().item())

      return summary


class MovingSymbols(torch.utils.data.Dataset):
    raw = None
    def __init__(self, root="/mnt/pccfs/not_backed_up/data/cube_data/spherecube_const_pitch_yaw_16", sequence_length=10, 
                  overfit=False, size=10000, train=False, height=16, width=16):
      super(MovingSymbols, self).__init__()

      import sys
      sys.path.append('./libs/moving-symbols/moving_symbols')
      from moving_symbols import MovingSymbolsEnvironment

      self.overfit = overfit
      self.batch_size = args.reader().batch_size

      self.params = {
        'data_dir':'./libs/moving-symbols/data/mnist',
        'split': 'training',
        'color_output': True,
        'symbol_labels': range(10),
        'position_speed_limits': (3, 3),
        'video_size': (height, width),
        'scale_limits': (.4, .4)
      }

      self.size = size if not self.overfit else self.batch_size

      raw = []
      from tqdm import tqdm
      for s in tqdm(range(self.size), desc='Building Dataset', leave=False):
        env = MovingSymbolsEnvironment(self.params, np.random.randint(0, 1))
        raw.append(np.array([np.asarray(env.next()) for _ in range(sequence_length)]) / 255.)
      raw = np.array(raw)

      self.data = torch.from_numpy(raw.astype(np.float32)).permute(0, 1, 4, 2, 3).contiguous()
      
      self.centering_const = self.data.mean()
      self.normalizing_const = self.data.abs().mean()

    def denormalize(self, x):
      return x * self.normalizing_const + self.centering_const

    def __getitem__(self, index):
      index = index % (self.size if not self.overfit else self.batch_size)

      raw = self.data[index]
      raw += (torch.rand_like(raw) - .5) * 1 / 256.
      raw = (raw - self.centering_const) / self.normalizing_const

      return raw, raw.view(-1)[0]

    def __len__(self):
      return self.size if not self.overfit else int(1e5)

    def __repr__(self):
      return ""

# class LTI2DSequence(torch.utils.data.Dataset):
#   def __init__(self, train=True, size=int(1e5), overfit=False, sequence_length=4, channels=4, state_dim=64 + 15, observation_dim=64):
#     self.__dict__.update(locals())

#     if overfit:
#       self.size = args.reader().batch_size

#     assert np.sqrt(observation_dim).is_integer(), 'observation_dim must be a perfect square. try {}'.format((int(np.sqrt(observation_dim)))**2)

#     rand_state = np.random.get_state()
#     np.random.seed(41)
#     self.state_dim = max(observation_dim, state_dim) * channels
#     self.observation_dim = observation_dim * channels
    
#     assert self.observation_dim <= self.state_dim, 'state_dim is too small'
    
#     self.A = np.random.randn(self.state_dim, self.state_dim)
#     u, s, v = np.linalg.svd(self.A)
#     self.A = u.dot(np.diag(1 + 0 * np.clip(.35 + np.random.rand(*s.shape), 0, 1))).dot(v)
#     self.A = np.eye(self.state_dim) * 1
#     u, s, v = np.linalg.svd(self.A)

#     self.C = np.zeros([self.observation_dim, self.state_dim])
#     self.C[:self.observation_dim, :self.observation_dim] = np.eye(self.observation_dim)
#     self.x0 = np.random.randn(size, self.state_dim)

#     # Normalize to unit length
#     self.x0 /= np.sqrt((self.x0 ** 2).sum(axis=1, keepdims=True))

#     # self.state_noise = np.random.randn(self.sequence_length, self.state_dim)

#     np.random.set_state(rand_state)

#   def __getitem__(self, idx):
#     obs = np.empty([self.sequence_length, self.observation_dim])
#     x = self.x0[idx]
#     for t in range(self.sequence_length):
#       x = self.A.dot(x)
#       y = self.C.dot(x)
#       obs[t] = y

#     k = int(np.sqrt(self.observation_dim / self.channels))
#     obs = obs.reshape(self.sequence_length, self.channels, k, k).astype(np.float32)

#     return torch.from_numpy(obs), torch.from_numpy(obs[-1])

#   def __len__(self):
#     return self.size
