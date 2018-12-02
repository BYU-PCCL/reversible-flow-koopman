import torch
import torch.nn as nn
import torch.nn.functional as F
from libs.args.args import argchoice, argignore
import numpy as np
import pdb
import utils

from modules import ReversibleFlow, Network

class EncodeStateDirectly(nn.Module):
  def __init__(self, dataset, extra_hidden_dim=8, l2_regularization=.001, flow:argchoice=[ReversibleFlow], inner_minimization_as_constant=True):
    self.__dict__.update(locals())
    super(EncodeStateDirectly, self).__init__()
    
    example = torch.stack([dataset[i][0][0] for i in range(20)])
    self.flow = flow(examples=example)
    self.observation_dim = torch.numel(example[0])
    self.example = example[0].unsqueeze(0)
    
    # round up to the nearest power of two, makes multi_matmul significantly faster
    self.hidden_dim = 1 << (self.observation_dim + extra_hidden_dim - 1).bit_length()

    self.U = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
    self.V = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
    self.alpha = nn.Parameter(torch.Tensor(self.hidden_dim))

    self.C = nn.Parameter(torch.Tensor(self.observation_dim, self.hidden_dim))

    self.reset_parameters()

  def __repr__(self):
    summary =  '                Name : {}\n'.format(self.__class__.__name__)
    summary += '    Hidden Dimension : {}\n'.format(self.hidden_dim)
    summary += 'Obervation Dimension : {}\n'.format(self.observation_dim)
    summary += '         A Dimension : {0}x{0} = {1}\n'.format(self.hidden_dim, self.hidden_dim**2)
    summary += '         C Dimension : {0}x{1} = {2}\n'.format(self.observation_dim, 
                                                               self.hidden_dim,
                                                               self.C.numel())
    return summary

  def reset_parameters(self):
    torch.nn.init.xavier_uniform_(self.C)
    torch.nn.init.xavier_uniform_(self.U)
    torch.nn.init.xavier_uniform_(self.V)

    # We want to initialize so that the singular values are near .9
    torch.nn.init.constant_(self.alpha, .90205)

  @utils.profile
  def multi_matmul_power_of_two(self, M):
    n = int(M.size(0))
    for i in range(n.bit_length() - 1):
      n = n >> 1
      M = torch.bmm(M[:n], M[n:])
    return M[0]

  def multi_matmul(self, M, left=None, right=None):
    if right - left == 1:
      return M[left]

    return torch.matmul(self.multi_matmul(M, left=(left + right)//2, right=right), 
                        self.multi_matmul(M, left=left, right=(left + right)//2))
  @utils.profile
  def build_A(self):
    Unorm = self.U / self.U.norm(dim=1, keepdim=True)
    Vnorm = self.V / self.V.norm(dim=1, keepdim=True)

    UH = torch.bmm(Unorm.unsqueeze(2), -2 * Unorm.unsqueeze(1))
    VH = torch.bmm(Vnorm.unsqueeze(2), -2 * Vnorm.unsqueeze(1))

    UH.reshape(UH.size(0), -1)[:, ::UH.size(1)+1] += 1
    VH.reshape(VH.size(0), -1)[:, ::VH.size(1)+1] += 1

    U = self.multi_matmul_power_of_two(UH)
    Vt = self.multi_matmul_power_of_two(VH)

    S = -torch.cos(self.alpha / 2.)
    A = U.matmul(S[None, :] * Vt)

    ##
    # The code above is eqivalent to and faster than the following:
    ##

    # I = torch.eye(self.hidden_dim, device=self.U.device)

    # Unorm = self.U / self.U.norm(dim=1, keepdim=True)
    # Vnorm = self.V / self.V.norm(dim=1, keepdim=True)

    # U, Vt = I, I
    # for k in range(self.hidden_dim):
    #   v = Unorm[k].unsqueeze(1)
    #   H = I - 2 * v.matmul(v.t())
    #   U = H.matmul(U)

    #   w = Vnorm[k].unsqueeze(1)
    #   H = I - 2 * w.matmul(w.t())
    #   Vt = H.matmul(Vt)

    # S = -torch.cos(self.alpha / 2.)

    # Atest = U.matmul(S[None, :] * Vt)

    # print(A - Atest)
    # exit()

    return A

  # @utils.profile
  # def build_O_M(self, sequence_length):
  #   M = torch.zeros((sequence_length * self.observation_dim, sequence_length * self.hidden_dim), device=torch.device('cuda'))
  #   T = self.C + 0. # the next line can't have T be a leaf node
  #   T.view(-1)[::self.C.size(1) + 1] += self.l2_regularization
  #   for offset in range(sequence_length):
  #     for i, j, in zip(range(sequence_length - offset), range(sequence_length - offset)):
  #       i = i + offset
  #       M[i * self.observation_dim:i * self.observation_dim + self.observation_dim, j * self.hidden_dim : j * self.hidden_dim + self.hidden_dim] = T
  #     T = T.matmul(self.A)
  
  #   return M[:, :self.hidden_dim], M[:, self.hidden_dim:]

  @utils.profile
  def build_O_A(self, sequence_length, device):
    A = self.build_A()
    O = [self.C]
    for t in range(sequence_length - 1):
       O.append(O[-1].matmul(A))
    O = torch.cat(O, dim=0)
    return O, A

  @utils.profile
  def forward(self, step, x, y):
    # fold sequence into batch
    xflat = x.reshape(-1, x.size(2), x.size(3), x.size(4))
    
    # encode each frame of each sequence in parallel
    glowloss, (zcat, zlist, logdet) = self.flow(step, xflat)
    #zcat, logdet = xflat, xflat.sum() * 0
 
    # unfold batch back into sequences
    # y is (batch, sequence, observation_dim)
    y = zcat.reshape(x.size(0), x.size(1), -1)
   
    # stack sequence into tall vector
    # Y is tall (batch, sequence * observation_dim) with stacked y's
    Y = y.reshape(y.size(0), -1)

    # Find x0*
    O, A = self.build_O_A(sequence_length=x.size(1), device=x.device)

    if self.inner_minimization_as_constant:
      # very fast, uses QR decomposition, not differentiable
      z, _ = torch.gels(Y.t(), O)
      x0 = z[:O.size(1)].detach()
    else:
      # This method is slower than using the QR decomposition, but torch.qr does not yet have gradients defined
      # see: https://github.com/pytorch/pytorch/blob/master/tools/autograd/derivatives.yaml#L618
      # to see if things have changed
      # Instead, we use the cholesky decomposition of O'O 
      U = torch.potrf(O.t().matmul(O), upper=False)
      rhs = Y.matmul(O)
      z = torch.trtrs(rhs.t(), U, transpose=False, upper=False)[0]
      x0 = torch.trtrs(z, U, transpose=True, upper=False)[0]

    Yhat = O.matmul(x0)
    w = Y.t() - Yhat

    wscaled = w.t().reshape(y.size()) #/ y.norm(dim=2, keepdim=True)

    prediction_error = .5 * (wscaled ** 2).mean()
    loss = prediction_error + (logdet.clamp(max=0)**2).mean() #+ (1 - x0.norm(dim=0).mean())**2

    return loss, (w, Y, Yhat, y, A, x0, logdet, prediction_error, zlist, zcat)

  @utils.profile
  def logger(self, step, data, out):
    x, yin = data
    b, s, c, h, w = x.shape
    w, Y, Yhat, y, A, x0, logdet, prediction_error, zlist, zcat = out
    stats = {}

    if step % 3 == 0:
      stats.update({'pred_err:prediction_error': prediction_error,
                    'logdet': logdet.clamp(max=0).mean()})

      stats.update({'gmax': max([p.grad.abs().max() for p in self.parameters() if p.grad is not None]),
                    'gnorm': max([p.grad.abs().norm() for p in self.parameters() if p.grad is not None])
                    })

    if step % 100 == 0:
      # TODO: log prediction and ground truth sequences
      stats.update({'|y|:mean_y_norm': y.norm(dim=2).mean(),
                    '|x|:mean_x_norm': x0.norm(dim=0).mean(),
                    'xₘₐₓ:x_max':x0.max()
                    })

    if step % 10**(int(np.log10(step + 1)) - 1) == 0:
      # heaven help me if I need to change how this indexing works
      # this is awkward, but we are converting shapes so that we go from
      # Yhat to Y to y to zcat to zlist then decoding and reshaping to compare with x
      zcat_hat = Yhat.t()[0:1].reshape(y[0:1].size()).reshape(-1, y[0:1].size(2))
      indexes = np.cumsum([0] + [m[:1*s].view(1*s, -1).size(1) for m in zlist])
      zlist_hat = [zcat_hat[:, a:b].reshape(c[:1*s].size()) for a, b, c in zip(indexes[:-1], indexes[1:], zlist)]
      xhat, _ = self.flow.decode(zlist_hat)

      def normalize(v):
        v = v - v.min()
        return v / v.max()

      recon_min = x[0][:, 0:1].min()
      shifted = x[0][:, 0:1] - recon_min
      recon_max = shifted.max()
      scaled = shifted / recon_max

      stats.update({'recon_loss': ((x[0] - xhat.reshape(x[0].size()))**2).mean(),
                    ':reconstruction': torch.clamp((xhat[:, 0:1] - recon_min) / recon_max, min=0, max=1),
                    ':truth': scaled})
      
      stats.update({':A': normalize(A.unsqueeze(0)),
                    ':C': normalize(self.C.unsqueeze(0)),
                    ':S': self.alpha})

    return stats

# if the y's scale down by 10, the x0 scales down by 10, and the loss scales down by 10
# how can we (properly) be agnostic to the scale of y?

# log it/s?
# print model and dataset summary at beginning
# large state spaces struggle to learn?
#############################
#############################
# RENAME EXPERIMENT
# FACTOR INTO TESTABLE HYPTOTHESIS
#############################
#############################

# actnorm isn't being initialized properly.. increase size of example batch

# YHAT OPTIONS
# yhat(k) = A^ky(0)
# yhat = OO+Y with O = [C, CA, CA^2, ..., CA^k]
# yhat(k) = p(yhat; Cxhat(k), Sigma(k)) using Kalman Smoother
# yhat(k) = OO+(Y - TE) with T = [C; CA C; CA^2 CA C; ...], e = [e0, e1, ... ek]

# LOSS OPTIONS
# loss = || y - yhat || + logdet
# loss = || y - yhat || + logdet + N(yhat; 0; I)
# loss = likelihood of yhat

# log sequences with prediction

