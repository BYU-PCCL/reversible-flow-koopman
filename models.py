import torch
import torch.nn as nn
import torch.nn.functional as F
from libs.args.args import argchoice, argignore
import numpy as np
import pdb
import utils

class Network(nn.Module):
  def __init__(self, hidden=512):
    self.__dict__.update(locals())
    super(Network, self).__init__()

  def lazy_init(self, x):
    if not hasattr(self, '_initialized'):
      self._initialized = True
      b, c, h, w = x.size()

      self.net = nn.Sequential(
        nn.Conv2d(c, self.hidden, 3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(self.hidden, self.hidden, 1, stride=1, padding=0),
        nn.ReLU(inplace=True),
        nn.Conv2d(self.hidden, c * 2, 3, stride=1, padding=1))

      # initalize with these
      self.net[-1].weight.data.fill_(0)
      self.net[-1].bias.data.fill_(0)

  def forward(self, x):
    self.lazy_init(x)
    h = self.net(x)
    shift = h[:, 0::2].contiguous()
    scale = h[:, 1::2].contiguous()

    return scale, shift

class ReversePermutation(nn.Module):
  def __init__(self):
    super(ReversePermutation, self).__init__()

  def forward(self, x, reverse=False):
    x1, x2 = torch.chunk(x, 2, dim=1)
    return torch.cat([x2, x1], dim=1)
    #return x.flip(1)

class NullPermutation(nn.Module):
  def __init__(self):
    super(NullPermutation, self).__init__()

  def forward(self, x, reverse=False):
    return x

class Squeeze(nn.Module):
  def __init__(self, factor=2):
    super(Squeeze, self).__init__()
    self.factor = factor

  def forward(self, x, reverse=False):
    b, c, h, w = x.size()
    if reverse:
      return F.fold(x.view(b, c, -1), (h * self.factor, w * self.factor), 
                kernel_size=(self.factor, self.factor), 
                stride=self.factor, dilation=1, padding=0)
    else:
      y = F.unfold(x, (self.factor, self.factor), 
                  stride=self.factor, dilation=1, padding=0)
      return y.view(b, -1, h // self.factor, w // self.factor)
    
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
    log_likelihood = -0.5 * (LogWhiteGaussian.log_2pi + ((z ** 2).view(z.size(0), -1).sum(1)))
    return log_likelihood

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
  def __init__(self, dataset, num_blocks=2, num_layers_per_block=10, squeeze_factor=2, 
               inner_var_cond=False, f:argchoice=[Network], permute:argchoice=[ReversePermutation, NullPermutation], 
               complex_log_frequency=2):

    self.__dict__.update(locals())
    super(ReversibleFlow, self).__init__()
    self.logprior = LogWhiteGaussian()
    self.networks = nn.ModuleList([nn.ModuleList([f() for _ in range(num_layers_per_block)]) 
                      for _ in range(num_blocks)])
    self.permutes = nn.ModuleList([nn.ModuleList([permute() for _ in range(num_layers_per_block)]) 
                      for _ in range(num_blocks)])

    self.prior = LogWhiteGaussian()
    self._logcount = 0

    c, h, w = dataset[0][0].size()
    self.squeezes = nn.ModuleList([Squeeze(factor=squeeze_factor) for _ in range(num_blocks)])
    assert (h / squeeze_factor**num_blocks) % 1 == 0, 'height {} is not divisible by {}^{}'.format(h, squeeze_factor, num_blocks)
    assert (w / squeeze_factor**num_blocks) % 1 == 0, 'width {} is not divisible by {}^{}'.format(w, squeeze_factor, num_blocks)
    
    # stoke lazy-initialization
    self.forward(dataset[0][0].unsqueeze(0), *dataset[0][1:])
    self.test(dataset[0][0].unsqueeze(0), *dataset[0][1:])

  @utils.profile
  def encode(self, x):
    loglikelihood_accum = 0

    # pad channels if not divisble by 2
    assert x.size(1) % 2 == 0, 'Input must have channels divisible by 2'

    zout = []
    z = x.clone()
    for L in range(self.num_blocks):
      # squeeze
      z = self.squeezes[L](z)

      for K in range(self.num_layers_per_block):
        # permute
        z = self.permutes[L][K](z)

        z1, z2 = torch.chunk(z, 2, dim=1)

        # step of flow
        scale, shift = self.networks[L][K](z1)
        safescale = torch.sigmoid(scale) * 2. + 0.01
        logscale = torch.log(safescale)
        z2 *= safescale
        z2 += shift

        inner_var_adjustment = (torch.log(z2.std()) + self.prior(z2.view(z2.size(0), -1)) * 1 / z2.std()) if self.inner_var_cond else 0
        loglikelihood_accum += logscale.view(logscale.size(0), -1).sum(1) - inner_var_adjustment

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

    zout = list(zout)
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
        safescale = torch.sigmoid(scale) * 2. + 0.01
        logscale = torch.log(safescale)
        z2 -= shift
        z2 /= safescale
        
        inner_var_adjustment = torch.log(z2.std()) + 1 / z2.std() if self.inner_var_cond else 0
        loglikelihood_accum -= logscale.view(logscale.size(0), -1).sum(1) - inner_var_adjustment

        z = torch.cat([z1, z2], dim=1)

        # unpermute
        z = self.permutes[L][K](z, reverse=True)

      # unsqueeze
      z = self.squeezes[L](z, reverse=True)

    return z, loglikelihood_accum

  def test(self, x, y):
    z, logdet = self.encode(x)
    xhat, logdet_zero = self.decode(z, logdet)
    assert (xhat - x).abs().max() < 1e-2, (xhat - x).abs().max().item()
    assert logdet_zero.abs().max() < 1e-1, logdet_zero.abs().max().item()
    #assert logdet.abs().sum() > 1e-3

  @utils.profile
  def forward(self, x, y):
    # x.normal_(0, 1)
    # x.fill_(0)
    # sx[:, 0, 0, 0].normal_()
    z, loglikelihood_accum = self.encode(x)
    zcat = torch.cat([x.view(x.size(0), -1) for x in z], dim=1)

    M = float(np.prod(x.shape[1:]))
    discretization_correction = float(np.log(2**8)) * M
    mean_log_likelihood = self.logprior(zcat) + loglikelihood_accum
    loss = (-mean_log_likelihood + discretization_correction) / M
    
    # the following command requires a sync operation
    # it should be as low in this function as possible
    loss = loss.mean() / float(np.log(2.))
    
    assert not np.isnan(loss.item()), 'loss is nan'
    
    return loss, (zcat, z, loglikelihood_accum)

  @utils.profile
  def log(self, data, step, out, **kwargs):
    self._logcount += 1
    x, y = data
    zcat, z, loglikelihood_accum = out

    # TODO: remove this ASAP
    stats = {}
    # stats.update({'gmax': max([p.grad.abs().max() for p in self.parameters()]),
    #               'gnorm': max([p.grad.abs().norm() for p in self.parameters()])})
    
    if self._logcount % self.complex_log_frequency == 0:

      stats = {'logdet': loglikelihood_accum.mean() / float(np.prod(x.shape[1:])),
                'mean': zcat.mean(),
                'std': zcat.std(),
                'var(std)': zcat.std(0).var()}

      # P = D.dot(D.T) / b
      # if not hasattr(self, '_running_cov_estimate'):
      #   self._running_estimate = np.eye(z.size(1))
      # self._running_estimate =  .20 * P + .80 * self._running_estimate

      try:
        D = zcat.t().cpu().detach().numpy()
        D -= D.mean(0, keepdims=True)
        k, b = D.shape
        B = D.T.dot(D) / k
        _, sv, _ = np.linalg.svd(B)
        stats.update({
                'stats.sv': [sv.min(), sv.max(), sv.mean(), sv.var()],
                'ld.sv': sv.var() / D.var()**2 / (float(b) / float(k)) - 1})
      except:
        pass

      # Test reconstruction accuracy
      xhat, _ = self.decode(z) # Needs to be last, since decode works "inplace" on the input array
      if (xhat - data[0]).abs().max() > 1e-2:
        print('Reconstruction Error:', (xhat - data[0]).abs().max())
        raise Exception('Reconstruction error is too high. Investigate.')

      sample_z = [m.clone().normal_() * m.std(0).unsqueeze(0) * .7 for m in z]
      [m[0].fill_(0) for m in sample_z]
      sample_images, _ = self.decode(sample_z)

      stats.update({#'kl_div(P)':  0.5 * (-np.linalg.slogdet(P)[1] - k + np.trace(P)),
              #'kl_div(B)':  0.5 * (-np.linalg.slogdet(B)[1] - b + np.trace(B)),
              #'cov': np.abs(self._running_estimate - np.eye(z.size(1))).mean(),
              'mean_image': sample_images[0, 0:1],
              'sample': sample_images[1:6 if sample_images.size(0) > 6 else sample_images.size(0), 0:1],
              'reconstructed': xhat[1:6 if zcat.size(0) > 6 else zcat.size(0), 0:1]
              #'z': zcat,
              #'zhist': zcat.view(-1)
              })

    return stats