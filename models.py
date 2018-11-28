import torch
import torch.nn as nn
import torch.nn.functional as F
from libs.args.args import argchoice, argignore
import numpy as np
import pdb
import utils

from modules import ReversibleFlow, Network

class EncodeStateDirectly(nn.Module):
  def __init__(self, dataset, extra_hidden_dim=0, l2_regularization=.001, flow:argchoice=[ReversibleFlow], inner_minimization_as_constant=True):
    self.__dict__.update(locals())
    super(EncodeStateDirectly, self).__init__()
    
    #self.flow = flow(example=dataset[0][0][0])
    self.observation_dim = torch.numel(dataset[0][0][0])
    self.hidden_dim = self.observation_dim + extra_hidden_dim

    self.A = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
    self.C = nn.Parameter(torch.Tensor(self.observation_dim, self.hidden_dim))

    self.reset_parameters()

  def reset_parameters(self):
    torch.nn.init.xavier_uniform_(self.A)
    torch.nn.init.xavier_uniform_(self.C)

    torch.nn.init.eye_(self.A)
    torch.nn.init.eye_(self.C)

  @utils.profile
  def build_O_M(self, sequence_length):
    M = torch.zeros((sequence_length * self.observation_dim, sequence_length * self.hidden_dim), device=torch.device('cuda'))
    T = self.C + 0. # the next line can't have T be a leaf node
    T.view(-1)[::self.C.size(1) + 1] += self.l2_regularization
    for offset in range(sequence_length):
      for i, j, in zip(range(sequence_length - offset), range(sequence_length - offset)):
        i = i + offset
        M[i * self.observation_dim:i * self.observation_dim + self.observation_dim, j * self.hidden_dim : j * self.hidden_dim + self.hidden_dim] = T
      T = T.matmul(self.A)
  
    return M[:, :self.hidden_dim], M[:, self.hidden_dim:]

  @utils.profile
  def build_O(self, sequence_length):
    O = torch.zeros((sequence_length * self.observation_dim, self.hidden_dim), device=torch.device('cuda'))
    T = self.C + 0. # the next line can't have T be a leaf node
    T.view(-1)[::self.C.size(1) + 1] += self.l2_regularization * 0.0
    for t in range(sequence_length):
      O[t * self.observation_dim:t * self.observation_dim + self.observation_dim] = T
      T = T.matmul(self.A)

    return O

  @utils.profile
  def forward(self, step, x, y):
    # fold sequence into batch
    xflat = x.reshape(-1, x.size(2), x.size(3), x.size(4))
    
    # encode each frame of each sequence in parallel
    #loss, (zcat, z, loglikelihood_accum) = self.flow(step, xflat)
    zcat = xflat.reshape(xflat.size(0), -1)
    
    # unfold batch back into sequences
    # y is (batch, sequence, observation_dim)
    y = zcat.reshape(x.size(0), x.size(1), -1)
   
    # stack sequence into tall vector
    # Y is tall (batch, sequence * observation_dim) with stacked y's
    Y = y.reshape(y.size(0), -1)

    # Find x0*
    O = self.build_O(sequence_length=x.size(1))

    if self.inner_minimization_as_constant:
      # very fast, uses QR decomposition, not differentiable
      z, _ = torch.gels(Y.t(), O)
      x0 = z[:O.size(1)].detach()
    else:
      # This method is slower than using the QR decomposition, but torch.qr does not yet have gradients defined
      # see: https://github.com/pytorch/pytorch/blob/master/tools/autograd/derivatives.yaml#L618
      # Instead, we use the cholesky decomposition of O'O 
      U = torch.potrf(O.t().matmul(O), upper=False)
      rhs = Y.matmul(O)
      z = torch.trtrs(rhs.t(), U, transpose=False, upper=False)[0]
      x0 = torch.trtrs(z, U, transpose=True, upper=False)[0]

    Yhat = O.matmul(x0)
    w = Y.t() - Yhat

    print(self.C.matmul(self.A))
    print(O)
    print(w)

    regularization_loss = self.A.abs().mean() + self.C.abs().mean()
    prediction_loss = w.abs().mean() #+ loss

    return prediction_loss, (w, Y, Yhat, y)

  @utils.profile
  def logger(self, step, data, out):
    w, Y, Yhat, y = out
    eigs, _ = torch.eig(self.A)

    normerror = torch.norm(y, dim=2) - torch.norm(Yhat.reshape(*y.shape), dim=2)
    normerror = normerror.abs().mean()

    print(torch.norm(y, dim=2)[0])
    print(torch.norm(Yhat.reshape(*y.shape), dim=2)[0])

    # TODO: log prediction and ground truth sequences
    return {'pred_err': w.abs().max(),
            '‖Y‖': Y.norm(),
            '‖∇A‖': self.A.grad.norm(),
            '‖∇C‖': self.C.grad.norm(),
            'λₘₐₓ': eigs[:, 0].abs().max(),
            'normerror': normerror,
            'A': self.A.unsqueeze(0),
            'C': self.C.unsqueeze(0)
            } #self.flow.logger(step, data, out)


# get batch of sequences
# encode each frame in sequence as y

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

