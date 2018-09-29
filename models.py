import torch
import torch.nn as nn
import torch.nn.functional as F
from libs.args.args import argchoice
import numpy as np
import pdb

class Network(nn.Module):
  def __init__(self):
    super(Network, self).__init__()

  def lazy_init(self, x):
    if not hasattr(self, '_initialized'):
      self._initialized = True
      b, c, h, w = x.size()

      self.net = nn.Sequential(
        nn.Conv2d(c, 10, 3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(10, 10, 1, stride=1, padding=0),
        nn.ReLU(inplace=True),
        nn.Conv2d(10, c * 2, 3, stride=1, padding=1))

      # initalize with these

  def forward(self, x):
    self.lazy_init(x)
    h = self.net(x)
    shift = h[:, 0::2]

    # the - 6 term corresponds to a bias of e^6, making
    # the derivative go to zero a little faster than linear
    # which improves stability
    scale = torch.log(h[:, 1::2].abs() + 2e-4)

    return scale, shift

class ReversePermutation(nn.Module):
  def __init__(self):
    super(ReversePermutation, self).__init__()

  def forward(self, x, reverse=False):
    return x.flip(1)

class Squeeze(nn.Module):
  def __init__(self, factor=2):
    super(Squeeze, self).__init__()
    self.factor = factor

  def forward(self, x, factor=2, reverse=False):
    b, c, h, w = x.size()
    if reverse:
      return F.fold(x.view(b, c, -1), (h * self.factor, w * self.factor), 
                kernel_size=(self.factor, self.factor), 
                stride=self.factor, dilation=1, padding=0)
    else:
      y = F.unfold(x, (self.factor, self.factor), 
                  stride=self.factor, dilation=1, padding=0)
      return y.view(b, -1, h // factor, w // factor)
    
class LogWhiteGaussian(nn.Module):
  log_2pi = float(np.log(2 * np.pi))

  def __init__(self):
    super(LogWhiteGaussian, self).__init__()
    self.mean = 0
    self.logs = 0

  def forward(self, z):
    log_likelihood = (-0.5 * (LogWhiteGaussian.log_2pi + 2. * self.logs + ((z - self.mean) ** 2).view(z.size(0), -1).mean(1) / np.exp(2. * self.logs)))
    return log_likelihood

# class FlowStep(Layer):
#     """
#         One step of flow described in paper
#                       ▲
#                       │
#         ┌─────────────┼─────────────┐
#         │  ┌──────────┴──────────┐  │
#         │  │ flow coupling layer │  │
#         │  └──────────▲──────────┘  │
#         │             │             │
#         │  ┌──────────┴──────────┐  │
#         │  │  flow permutation   │  │
#         │  │        layer        │  │
#         │  └──────────▲──────────┘  │
#         │             │             │
#         │  ┌──────────┴──────────┐  │
#         │  │     activation      │  │
#         │  │ normalization layer │  │
#         │  └──────────▲──────────┘  │
#         └─────────────┼─────────────┘
#                       │
#                       │
#     """

class ReversibleFlow(nn.Module):
  """
        Flow model with multi-scale architecture
                         ┏━━━┓
                         ┃z_L┃
                         ┗━▲━┛
                           │
                ┌──────────┴──────────┐
                │    step of flow     │* K
                └──────────▲──────────┘
                ┌──────────┴──────────┐
                │       squeeze       │
                └──────────▲──────────┘
                           ├──────────────┐
        ┏━━━┓   ┌──────────┴──────────┐   │
        ┃z_i┃◀──┤        split        │   │
        ┗━━━┛   └──────────▲──────────┘   │
                ┌──────────┴──────────┐   │
                │    step of flow     │* K│ * (L-1)
                └──────────▲──────────┘   │
                ┌──────────┴──────────┐   │
                │       squeeze       │   │
                └──────────▲──────────┘   │
                           │◀─────────────┘
                         ┏━┻━┓
                         ┃ x ┃
                         ┗━━━┛
  """
  def __init__(self, dataset, num_blocks=1, num_layers_per_block=2, squeeze_factor=2, f:argchoice=[Network], permute:argchoice=[ReversePermutation]):
    self.__dict__.update(locals())
    super(ReversibleFlow, self).__init__()
    self.prior = LogWhiteGaussian()
    self.networks = nn.ModuleList([nn.ModuleList([f() for _ in range(num_layers_per_block)]) 
                      for _ in range(num_blocks)])
    self.permutes = nn.ModuleList([nn.ModuleList([permute() for _ in range(num_layers_per_block)]) 
                      for _ in range(num_blocks)])

    c, h, w = dataset[0][0].size()
    self.squeezes = nn.ModuleList([Squeeze(factor=squeeze_factor) for _ in range(num_blocks)])
    assert (h / squeeze_factor**num_blocks) % 1 == 0, 'height {} is not divisible by {}^{}'.format(h, squeeze_factor, num_blocks)
    assert (w / squeeze_factor**num_blocks) % 1 == 0, 'width {} is not divisible by {}^{}'.format(w, squeeze_factor, num_blocks)
    
    # stoke lazy-initialization
    self.forward(dataset[0][0].unsqueeze(0), *dataset[0][1:])
    self.test(dataset[0][0].unsqueeze(0), *dataset[0][1:])

  def encode(self, x):
    logdet_accum = 0

    # pad channels if not divisble by 2
    if x.size(1) % 2 != 0:
      x = torch.cat([x, x[:, 0:1]], dim=1)

    z = x
    for L in range(self.num_blocks):
      # squeeze
      z = self.squeezes[L](z)

      for K in range(self.num_layers_per_block):
        # permute
        z = self.permutes[L][K](z)

        # step of flow
        z1, z2 = torch.chunk(z, 2, dim=1)
        logscale, shift = self.networks[L][K](z1)
        z2 += shift
        z2 *= torch.exp(logscale)

        logdet_accum += logscale.view(logscale.size(0), -1).mean(1)
        z = torch.cat([z1, z2], dim=1)

      # split heriachical 

    return z, logdet_accum

  def decode(self, z, logdet_input_accum=0):

    # all the operation on logdet_accum are in-place, but it's not natural to think
    # of arguments as being pass-by-reference in python, so we create a seperate node
    logdet_accum = logdet_input_accum + 0

    for L in reversed(range(self.num_blocks)):
      # reverse squeeze
      for K in reversed(range(self.num_layers_per_block)):
        #reverse flow step
        z1, z2 = torch.chunk(z, 2, dim=1)
        logscale, shift = self.networks[L][K](z1)
        z2 /= torch.exp(logscale)
        z2 -= shift
        logdet_accum -= logscale.view(logscale.size(0), -1).mean(1)
        z = torch.cat([z1, z2], dim=1)

        # unpermute
        z = self.permutes[L][K](z, reverse=True)

      #unsqueeze
      z = self.squeezes[L](z, reverse=True)

    return z, logdet_accum

  def test(self, x, y):
    z, logdet = self.encode(x)
    xhat, logdet_zero = self.decode(z, logdet)
    assert (xhat - x).abs().max() < 1e-3, (xhat - x).abs().sum().item()
    assert logdet_zero.abs().max() < 1e-3, logdet_zero.abs().sum().item()
    #assert logdet.abs().sum() > 1e-3

  def forward(self, x, y):
    x = x.normal_()
    z, logdet_accum = self.encode(x)

    n_bins = 2**8
    M = np.prod(x.shape[1:])
    discretization_correction = float(np.log(n_bins))
    mean_log_likelihood = self.prior(z) + logdet_accum

    loss = -mean_log_likelihood #+ discretization_correction
    loss /=  float(np.log(2.))
    loss = loss.mean()

    return loss, z

  def log(self, step, out, **kwargs):

    if step % 5== 0:
      return {'frob_norm_cov': ((np.eye(1568) - np.cov(out.view(out.size(0), -1).t().cpu().detach().numpy()))**2).sum() - 1568}

    return {}