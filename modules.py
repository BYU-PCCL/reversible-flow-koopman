import torch
import torch.nn as nn
import torch.nn.functional as F
from libs.args.args import argchoice, argignore
import numpy as np
import pdb
import utils

import torch.utils.checkpoint as checkpoint
import einops

class Network(nn.Module):
  def __init__(self, hidden=512):
    self.__dict__.update(locals())
    super(Network, self).__init__()

    checkpoint.preserve_rng_state = False

  def lazy_init(self, x):
    if not hasattr(self, '_initialized'):
      self._initialized = True
      b, c, h, w = x.size()


      self.net = nn.Sequential(
        nn.Conv2d(c, self.hidden, 3, stride=1, padding=1),
        nn.LeakyReLU(inplace=True),
        #nn.Dropout2d(p=0.3),
        nn.Conv2d(self.hidden, self.hidden, 1, stride=1, padding=0),
        nn.LeakyReLU(inplace=True),
        #nn.Dropout2d(p=0.3),
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

class LinearNetwork(Network):
   def lazy_init(self, x):
    if not hasattr(self, '_initialized'):
      self._initialized = True
      b, c, h, w = x.size()

      self.net = nn.Sequential(
        nn.Conv2d(c, self.hidden, 5, stride=1, padding=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(self.hidden, self.hidden, 1, stride=1, padding=0),
        nn.ReLU(inplace=True),
        nn.Conv2d(self.hidden, c * 2, 5, stride=1, padding=2))

      # initalize with these
      self.net[-1].weight.data.fill_(0)
      self.net[-1].bias.data.fill_(0)

class ReversePermutation(nn.Module):
  def __init__(self):
    super(ReversePermutation, self).__init__()

  def forward(self, x, reverse=False):
    x1, x2 = torch.chunk(x, 2, dim=1)
    x = torch.cat([x2, x1], dim=1)
    return x, 0
    #return x.flip(1)

class NullPermutation(nn.Module):
  def __init__(self):
    super(NullPermutation, self).__init__()

  def forward(self, x, reverse=False):
    return x, 0

class Unitary(nn.Module):
  def __init__(self, dim, fast=True):
    self.__dict__.update(locals())
    super(Unitary, self).__init__()

    if fast:
      assert ((dim & (dim - 1)) == 0) and dim > 0, 'dim must be a power of two for fast computation'

    self.U = nn.Parameter(torch.Tensor(dim, dim))
    self.reset_parameters()
  
  def reset_parameters(self):
    torch.nn.init.xavier_uniform_(self.U)
    self.U.data /= self.U.norm(dim=1, keepdim=True)

    self.U.data *= 0
    self.U.data.view(-1)[::self.U.size(1) + 1] += 1

  def multi_matmul(self, M, left=None, right=None):
    left = left or 0
    right = right or len(M)

    if right - left == 1:
      return M[left]

    return torch.matmul(self.multi_matmul(M, left=(left + right)//2, right=right), 
                        self.multi_matmul(M, left=left, right=(left + right)//2))

  def multi_matmul_power_of_two(self, M):
    n = int(M.size(0))
    for i in range(n.bit_length() - 1):
      n = n >> 1
      M = torch.bmm(M[:n], M[n:])
    return M[0]

  def get(self):
    Unorm = self.U / self.U.norm(dim=1, keepdim=True)
    UH = torch.bmm(Unorm.unsqueeze(2), -2 * Unorm.unsqueeze(1))
    UH.reshape(UH.size(0), -1)[:, ::UH.size(1)+1] += 1

    if self.fast:
      U = self.multi_matmul_power_of_two(UH)
    else:
      U = self.multi_matmul(UH)
      
    return U

class Orthogonal1x1Conv(nn.Module):
  def __inti__(self):
    super(Orthogonal1x1Conv, self).__init__()

  def lazy_init(self, x):
    if not hasattr(self, '_initialized'):
      self._initialized = True
      dim = x.shape[1]
      self.W = Unitary(dim, fast=((dim & (dim - 1)) == 0) and dim > 0)

  def forward(self, x, reverse=False):
    self.lazy_init(x)
    weight = self.W.get().t() if reverse else self.W.get()
    z = F.conv2d(x, weight.unsqueeze(2).unsqueeze(2))
    return z, 0

class Invertible1x1Conv(nn.Module):
  def __init__(self, LU_decomposed=False):
    super().__init__()
    self.LU = LU_decomposed

  def lazy_init(self, x):
    if not hasattr(self, '_initialized'):
      self._initialized = True
      num_channels = x.size(1)
      w_shape = [num_channels, num_channels]
      w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
      if not self.LU:
        # Sample a random orthogonal matrix:
        self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
      else:
        np_p, np_l, np_u = scipy.linalg.lu(w_init)
        np_s = np.diag(np_u)
        np_sign_s = np.sign(np_s)
        np_log_s = np.log(np.abs(np_s))
        np_u = np.triu(np_u, k=1)
        l_mask = np.tril(np.ones(w_shape, dtype=np.float32), -1)
        eye = np.eye(*w_shape, dtype=np.float32)

        self.p = torch.Tensor(np_p.astype(np.float32))
        self.sign_s = torch.Tensor(np_sign_s.astype(np.float32))
        self.l = nn.Parameter(torch.Tensor(np_l.astype(np.float32)))
        self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(np.float32)))
        self.u = nn.Parameter(torch.Tensor(np_u.astype(np.float32)))
        self.l_mask = torch.Tensor(l_mask)
        self.eye = torch.Tensor(eye)
      self.w_shape = w_shape

  def get_weight(self, input, reverse):
      w_shape = self.w_shape
      if not self.LU:
        pixels = np.prod(input.shape[-2:])
        dlogdet = torch.slogdet(self.weight)[1] * pixels
        if not reverse:
            weight = self.weight.view(w_shape[0], w_shape[1], 1, 1)
        else:
            weight = torch.inverse(self.weight.double()).float()\
                          .view(w_shape[0], w_shape[1], 1, 1)
        return weight, dlogdet
      else:
        self.p = self.p.to(input.device)
        self.sign_s = self.sign_s.to(input.device)
        self.l_mask = self.l_mask.to(input.device)
        self.eye = self.eye.to(input.device)
        l = self.l * self.l_mask + self.eye
        u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(self.log_s))
        dlogdet = thops.sum(self.log_s) * thops.pixels(input)
        if not reverse:
          w = torch.matmul(self.p, torch.matmul(l, u))
        else:
          l = torch.inverse(l.double()).float()
          u = torch.inverse(u.double()).float()
          w = torch.matmul(u, torch.matmul(l, self.p.inverse()))
        return w.view(w_shape[0], w_shape[1], 1, 1), dlogdet

  def forward(self, input, reverse=False):
      """
      log-det = log|abs(|W|)| * pixels
      """
      self.lazy_init(input)

      weight, dlogdet = self.get_weight(input, reverse)
      z = F.conv2d(input, weight)
      return z, dlogdet

class Squeeze(nn.Module):
  def __init__(self, factor=2):
    super(Squeeze, self).__init__()
    self.factor = factor

  def forward(self, x, reverse=False):
    # if reverse:
    #   return einops.rearrange(x, 'b (c h2 w2) h w -> b c (h h2) (w w2)', h2=self.factor, w2=self.factor)
    # else:
    #   return einops.rearrange(x, 'b c (h h2) (w w2) -> b (c h2 w2) h w', h2=self.factor, w2=self.factor)
    
    ##
    # the above is slightly faster, but equivalent to the following:
    ##
    b, c, h, w = x.size()
    if reverse:
      return F.fold(x.view(b, c, -1), (h * self.factor, w * self.factor), 
                kernel_size=(self.factor, self.factor), 
                stride=self.factor, dilation=1, padding=0)
    else:
      x = F.unfold(x, (self.factor, self.factor), 
                       stride=self.factor, dilation=1, padding=0)
      x = x.reshape(b, -1, h // self.factor, w // self.factor)
      return x
    
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
    # TBH I'm not sure why we don't have the n (number of dimensions)
    # in this formula... but that's how the Glow code did it
    log_likelihood = -0.5 * (LogWhiteGaussian.log_2pi + ((z ** 2)))
    return log_likelihood.view(z.size(0), -1).sum(1)

class Clamp(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input, min, max):
    ctx.save_for_backward(input, max)
    return torch.min(input, max)

  @staticmethod
  def backward(ctx, grad_output):
    input, max = ctx.saved_tensors
    return torch.where((input > max) & (grad_output < 0),  grad_output * 0 + 10, grad_output), None, None

class SigmoidShiftScaler(nn.Module):
  def __init__(self, b=0.001, m=2):
    self.__dict__.update(locals())
    super(SigmoidShiftScaler, self).__init__()
    self.sigmoid_shift = np.log((1 - self.b) / (self.m - 1))

  def forward(self, scale, z):
    return torch.sigmoid(scale + self.sigmoid_shift) * (self.m - self.b) + self.b

class GlowShift(nn.Module):
  def forward(self, scale, z):
    return torch.sigmoid(scale + 2)

class ClampScaler(nn.Module):
  def __init__(self, b=0.001, m=2):
    self.__dict__.update(locals())
    super(ClampScaler, self).__init__()

  def forward(self, scale, z):
    return torch.min(scale, self.m / (torch.abs(z.detach()) + self.b))

class FreeScaler(nn.Module):
  def __init__(self, b=0.001, m=2):
    self.__dict__.update(locals())
    super(FreeScaler, self).__init__()

  def forward(self, scale, z):
    return (scale + 1 - self.b).abs() + self.b 

class AdditiveOnlyShiftScaler(nn.Module):
  def __init__(self, c=1):
    self.__dict__.update(locals())
    super(AdditiveOnlyShiftScaler, self).__init__()

  def forward(self, scale, z):
    return scale * 0 + self.c

class ActNorm(nn.Module):
  def lazy_init(self, x):
    if not hasattr(self, '_initialized'):
      self._initialized = True

      b, c, h, w = x.size()
      self.bias = nn.Parameter(torch.zeros(1, c, 1, 1))
      self.logs = nn.Parameter(torch.zeros(1, c, 1, 1))

      bias = -x.sum(dim=[0, 2, 3], keepdim=True) / (b * h * w)
      variance = ((x + bias) ** 2).sum(dim=[0, 2, 3], keepdim=True) / (b * h * w)
      logs = -torch.log(torch.sqrt(variance) + 1e-6)
      self.bias.data.copy_(bias.data)
      self.logs.data.copy_(logs.data)

  def forward(self, x, reverse=False):
    self.lazy_init(x)

    b, c, h, w = x.size()

    if not reverse:
      x += self.bias
      x *= torch.exp(self.logs)
    else:
      x /= torch.exp(self.logs)
      x -= self.bias

    logdet = self.logs.sum(dim=[1,2,3]) * (h * w)
    return x, logdet

class AffineFlowStep(nn.Module):
  def __init__(self, f:argchoice=[Network, LinearNetwork], 
                     actnorm=False,
                     safescaler:argchoice=[SigmoidShiftScaler, AdditiveOnlyShiftScaler, ClampScaler, FreeScaler, GlowShift]):
    self.__dict__.update(locals())
    super(AffineFlowStep, self).__init__()
    self.f = f()
    self.safescaler = safescaler()
    self.actnorm = ActNorm() if self.actnorm else None

  @utils.profile
  def forward(self, z, reverse=False, K=None):
    z1, z2 = torch.chunk(z, 2, dim=1)

    scale, shift = self.f(z1)
    safescale = self.safescaler(scale, z2)
    logscale = torch.log(safescale)
    logdet = logscale.view(logscale.size(0), -1).sum(1)

    # if np.random.rand() < .01:
    #   print('safescale', safescale.mean().item(), logscale.mean())

    if not reverse:
      if self.actnorm:
        z2, logdet_actnorm = self.actnorm(z2)
        logdet += logdet_actnorm

      z2 = (z2 + shift) * safescale

    else:
      z2 = (z2 / safescale) - shift

      if self.actnorm:
        z2, logdet_actnorm = self.actnorm(z2, reverse=True)
        logdet += logdet_actnorm

    z = torch.cat([z1, z2], dim=1)

    return z, logdet

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
  # 3blocks 32layers
  def __init__(self, examples, num_blocks=4, num_layers_per_block=10, squeeze_factor=2, 
               inner_var_cond=False, 
               num_projections=10,
               prior:argchoice=[LogWhiteGaussian],
               flow:argchoice=[AffineFlowStep], 
               permute:argchoice=[ReversePermutation, NullPermutation, Invertible1x1Conv, Orthogonal1x1Conv],
               checkpoint_gradients=False):

    self.__dict__.update(locals())
    super(ReversibleFlow, self).__init__()
    self.logprior = prior()
    
    self.permutes = nn.ModuleList([nn.ModuleList([permute() for _ in range(num_layers_per_block)]) 
                      for _ in range(num_blocks)])
    self.flows = nn.ModuleList([nn.ModuleList([flow() for _ in range(num_layers_per_block)]) 
                      for _ in range(num_blocks)])
    self.squeezes = nn.ModuleList([Squeeze(factor=squeeze_factor) for _ in range(num_blocks)])

    # A little test ot see if we will run into any problems
    b, c, h, w = examples.size()
    
    assert (h / squeeze_factor**num_blocks) % 1 == 0, 'height {} is not divisible by {}^{}'.format(h, squeeze_factor, num_blocks)
    assert (w / squeeze_factor**num_blocks) % 1 == 0, 'width {} is not divisible by {}^{}'.format(w, squeeze_factor, num_blocks)
    
    # stoke lazy-initialization
    _, (z, _, _) = self.forward(-1, examples)
    self.test(examples)

    #self.register_buffer('projections', torch.zeros(z.size(1), num_projections))

  @utils.profile
  def encode(self, x):
    loglikelihood_accum = 0

    zout = []
    z = x.clone()
    z.requires_grad = True

    for L in range(self.num_blocks):
      # squeeze - ensures the channel dimension is divisible by 2
      z = self.squeezes[L](z)

      for K in range(self.num_layers_per_block):
        # permute
        z, plogdet = self.permutes[L][K](z)

        if self.checkpoint_gradients:
          z, logdet = utils.checkpoint(self.flows[L][K], z)
        else:
          z, logdet = self.flows[L][K](z)
          
        loglikelihood_accum = loglikelihood_accum + (logdet + plogdet)

        del plogdet
        del logdet
 
      # split hierarchical 
      # this operation returns two non-contiguous blocks
      # with references to the original z tensor
      # if we do not call .contiguous() (or .clone()) on BOTH z1 and z2, 
      # then the entire z tensor must be kept around
      # the del operators just do a little cleanup to avoid a (very) slight memory bump
      z1, z2 = torch.chunk(z, 2, dim=1)
      z1 = z1.contiguous()
      z2 = z2.contiguous()
      zout.append(z1)
      del z
      z = z2
      del z2

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
        
        z, logdet = self.flows[L][K](z, reverse=True)

        # unpermute
        z, plogdet = self.permutes[L][K](z, reverse=True)

        loglikelihood_accum -= logdet + plogdet

      # unsqueeze
      z = self.squeezes[L](z, reverse=True)

    return z, loglikelihood_accum

  def test(self, x):
    self.eval()
    with torch.no_grad():
      z, logdet = self.encode(x)
      xhat, logdet_zero = self.decode(z, logdet)
      assert (xhat - x).abs().max() < 1.5e-5, (xhat - x).abs().max().item()
      assert logdet_zero.abs().max() < 1e-1, logdet_zero.abs().max().item()
    self.train()

  def forward(self, step, x, loss=True):

    z, loglikelihood_accum = self.encode(x)

    zcat = torch.cat([m.reshape(m.size(0), -1) for m in z], dim=1)

    if loss:
      M = float(np.prod(x.shape[1:]))
      discretization_correction = float(-np.log(256)) * M
      log_likelihood = self.logprior(zcat) + loglikelihood_accum #+ discretization_correction
      loss = -log_likelihood
      
      # the following command requires a sync operation
      # it should be as low in this function as possible
      loss = loss.mean() / float(np.log(2.) * M)
      #assert not np.isnan(loss.item()), 'loss is nan'
    
    return loss, (zcat, z, loglikelihood_accum)

  def logger(self, step, data, out):
    x, y = data
    zcat, z, loglikelihood_accum = out

    stats = {}
    stats.update({'logdet': loglikelihood_accum.mean() / float(np.prod(x.shape[1:])),
              'mean': zcat.mean(),
              'std': zcat.std(),
              'var(std)': zcat.std(0).var()})

    # P = D.dot(D.T) / b
    # if not hasattr(self, '_running_cov_estimate'):
    #   self._running_estimate = np.eye(z.size(1))
    # self._running_estimate =  .20 * P + .80 * self._running_estimate
    if step % 10 == 0:

      self.projections.random_()
      self.projections /= self.projections.norm(dim=0, keepdim=True)

      K = zcat.mm(self.projections)
      K -= K.mean(0)
      K /= K.std(0)

      # http://mathworld.wolfram.com/k-Statistic.html
      n = K.size(0)
      s1, s2, s3, s4 = [(K ** i).sum(0) for i in range(1, 5)]
      k3 = (2 * s1**3 - 3*n*s1 * s2 + n**2 * s3) / (n * (n - 1) * (n - 2))
      k4 = ((-6 * s1**4 + 12 * n * s1**2 * s2 - 3 * n * (n - 1) * s2 ** 2 - 4 * n * (n + 1) * s1 * s3 + n**2 * (n + 1) * s4)) / (n * (n - 1) * (n - 2) * (n - 3))
      
      #Eq. 8 of https://www.jstor.org/tc/accept?origin=%2Fstable%2Fpdf%2F2981662.pdf
      q = (k3**2 + .25 * k4**2) / 12

      stats.update({'q': q.mean(),
                    'qmax': q.max(),
                    'qvar': q.var()})

      stats.update({'gmax': max([p.grad.abs().max() for p in self.parameters()]),
                    'gnorm': max([p.grad.abs().norm() for p in self.parameters()])})
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
      # xhat, _ = self.decode(z) # Needs to be last, since decode works "inplace" on the input array
      # if (xhat - data[0]).abs().max() > 1e-2:
      #   print('Reconstruction Error:', (xhat - data[0]).abs().max())
      #   raise Exception('Reconstruction error is too high. Investigate.')

      V = [v[:min(6, v.size(0))] for v in z]
      sample_z = [v.clone().normal_() * v.std(0).unsqueeze(0) for v in V]
      [v[0].fill_(0) for v in sample_z]
      sample_images, _ = self.decode(sample_z)

      stats.update({
              'mean_image': sample_images[0, 0:1],
              'sample': sample_images[1:, 0:1],
              })

    return stats
