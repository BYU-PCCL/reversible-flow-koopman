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
      self.net[-1].weight.data.fill_(0)
      self.net[-1].bias.data.fill_(0)

  def forward(self, x):
    self.lazy_init(x)
    h = self.net(x)
    shift = h[:, 0::2].contiguous()
    scale = h[:, 1::2].contiguous()

    return scale, shift

# https://github.com/chaiyujin/glow-pytorch/blob/master/glow/modules.py
# class ActNorm(nn.Module):
#   """
#   Activation Normalization
#   Initialize the bias and scale with a given minibatch,
#   so that the output per-channel have zero mean and unit variance for that.
#   After initialization, `bias` and `logs` will be trained as parameters.
#   """

#   def __init__(self, num_features, scale=1.):
#       super(ActNorm).__init__()
#       self.scale = scale

#   def lazy_init(self, input):
#     b, c, h, w = input.size()
#     if not hasattr(self, '_initialized'):
#       self._initialized = True
#       # register mean and scale
#       size = [1, c, 1, 1]
#       self.bias = nn.Parameter(torch.zeros(*size))
#       self.logs = nn.Parameter(torch.zeros(*size))

#       with torch.no_grad():
#         bias = thops.mean(input.clone(), dim=[0, 2, 3], keepdim=True) * -1.0
#         vars = thops.mean((input.clone() + bias) ** 2, dim=[0, 2, 3], keepdim=True)
#         logs = torch.log(self.scale/(torch.sqrt(vars)+1e-6))
#         self.bias.data.copy_(bias.data)
#         self.logs.data.copy_(logs.data)
#         self.inited = True

#   def forward(self, input, logdet=None, reverse=False):
#     self.lazy_init(input)
#     if not reverse:
#         input += self.bias
#         input *= torch.exp(self.logs)
#         logdet += self.logs.sum() / np.prod(self.logs.shape[2:])
#     else:
#         # scale and center
#         input /= torch.exp(self.logs)
#         input -= self.bias
#         logdet -= self.logs.sum() / np.prod(self.logs.shape[2:])
#     return input, logdet

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
    
class LogGaussian(nn.Module):
  log_2pi = float(np.log(2 * np.pi))

  def __init__(self, mean, logs):
    self.__dict__.update(locals())
    super(LogWhiteGaussian, self).__init__()

  def forward(self, z):
    log_likelihood = (-0.5 * (LogWhiteGaussian.log_2pi + 2. * self.logs + ((z - self.mean) ** 2).view(z.size(0), -1).mean(1) / np.exp(2. * self.logs)))
    return log_likelihood

class LogWhiteGaussian(nn.Module):
  log_2pi = float(np.log(2 * np.pi))

  def __init__(self, mean=0, logs=0):
    super(LogWhiteGaussian, self).__init__()

  def forward(self, z):
    log_likelihood = -0.5 * (LogWhiteGaussian.log_2pi + ((z ** 2).view(z.size(0), -1).mean(1)))
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
  def __init__(self, dataset, num_blocks=1, num_layers_per_block=1, squeeze_factor=2, inner_var_cond=True, f:argchoice=[Network], permute:argchoice=[ReversePermutation], kl_div_log_frequency=1000):
    self.__dict__.update(locals())
    super(ReversibleFlow, self).__init__()
    self.logprior = LogWhiteGaussian()
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
    loglikelihood_accum = 0

    # pad channels if not divisble by 2
    assert x.size(1) % 2 == 0, 'Input must have channels divisible by 2'

    zout = []
    z = x
    for L in range(self.num_blocks):
      # squeeze
      z = self.squeezes[L](z)

      for K in range(self.num_layers_per_block):
        # permute
        z = self.permutes[L][K](z)

        # step of flow
        z1, z2 = torch.chunk(z, 2, dim=1)
        scale, shift = self.networks[L][K](z1)
        z2 += shift
        z2 *= torch.exp(scale)

        inner_var_adjustment = torch.log(z2.std()) + 1 / z2.std() if self.inner_var_cond else 0
        loglikelihood_accum += scale.view(scale.size(0), -1).mean(1) - inner_var_adjustment

        z = torch.cat([z1, z2], dim=1)

      # split hierarchical 
      z1, z2 = torch.chunk(z, 2, dim=1)
      zout.append(z1)
      z = z2

    zout.append(z)

    return zout, loglikelihood_accum

  def decode(self, zout, log_likelihood=0):

    # all the operation on loglikelihood_accum are in-place, but it's not natural to think
    # of arguments as being pass-by-reference in python, so we create a seperate node
    loglikelihood_accum = log_likelihood + 0

    z = zout.pop()

    for L in reversed(range(self.num_blocks)):
      # combine hierarchical
      z2 = z
      z1 = zout.pop()
      z = torch.cat([z1, z2], dim=1)

      # reverse squeeze
      for K in reversed(range(self.num_layers_per_block)):
        #reverse flow step
        z1, z2 = torch.chunk(z, 2, dim=1)

        scale, shift = self.networks[L][K](z1)
        z2 /= torch.exp(scale)
        z2 -= shift

        inner_var_adjustment = torch.log(z2.std()) + 1 / z2.std() if self.inner_var_cond else 0
        loglikelihood_accum -= scale.view(scale.size(0), -1).mean(1) - inner_var_adjustment

        z = torch.cat([z1, z2], dim=1)

        # unpermute
        z = self.permutes[L][K](z, reverse=True)

      # unsqueeze
      z = self.squeezes[L](z, reverse=True)

    return z, loglikelihood_accum

  def test(self, x, y):
    z, logdet = self.encode(x)
    xhat, logdet_zero = self.decode(z, logdet)
    assert (xhat - x).abs().max() < 1e-1, (xhat - x).abs().max().item()
    assert logdet_zero.abs().max() < 1e-1, logdet_zero.abs().max().item()
    #assert logdet.abs().sum() > 1e-3

  def forward(self, x, y):
    #x.normal_(0, 2)
    z, loglikelihood_accum = self.encode(x)

    # flatten all the hierarchical layers
    z = torch.cat([x.view(x.size(0), -1) for x in z], dim=1)

    n_bins = 2**8
    M = np.prod(x.shape[1:])
    discretization_correction = float(np.log(n_bins))
    mean_log_likelihood = self.logprior(z) + .5 * loglikelihood_accum

    loss = -mean_log_likelihood #+ discretization_correction
    loss /=  float(np.log(2.))
    loss = loss.mean()

    return loss, (z, loglikelihood_accum)

  def log(self, step, out, **kwargs):
    z, loglikelihood_accum = out
    if step % self.kl_div_log_frequency == 0:
      D = z.view(z.size(0), -1).t().cpu().detach().numpy()
      k, b = D.shape
      P = D.dot(D.T) / b

      kl_div = 0.5 * (-np.linalg.slogdet(P)[1] - k + np.trace(P))

      return {'kl_div': kl_div}

    grad = self.networks[0][0].net[0].weight.grad.abs().mean()

    return {'logdet': loglikelihood_accum.mean(),
            'mean': z.mean(),
            'std': z.std(),
            'early_grad': grad}