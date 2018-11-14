import torch
import torch.nn as nn
import torch.nn.functional as F
from libs.args.args import argchoice, argignore
import numpy as np
import pdb
import utils

from modules import ReversibleFlow, Network

class EncodeStateDirectly(nn.Module):
  def __init__(self, dataset, extra_hidden_dim=20, l2_regularization=.001, flow:argchoice=[ReversibleFlow]):
    self.__dict__.update(locals())
    super(EncodeStateDirectly, self).__init__()
    
    self.flow = flow(example=dataset[0][0][0])
    self.observation_dim = torch.numel(dataset[0][0][0])
    self.hidden_dim = self.observation_dim + extra_hidden_dim

    self.A = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
    self.C = nn.Parameter(torch.Tensor(self.observation_dim, self.hidden_dim))

    self.reset_parameters()

  def reset_parameters(self):
    torch.nn.init.xavier_uniform_(self.A)
    torch.nn.init.xavier_uniform_(self.C)

  def build_M(self, sequence_length):
    M = torch.zeros((sequence_length * self.observation_dim, sequence_length * self.hidden_dim), device=torch.device('cuda'))
    T = self.C + 0. # the next line can't have T be a leaf node
    T.view(-1)[::self.C.size(1) + 1] += self.l2_regularization

    for offset in range(sequence_length):
      for i, j, in zip(range(sequence_length - offset), range(sequence_length - offset)):
        i = i + offset
        M[i * self.observation_dim:i * self.observation_dim + self.observation_dim, j * self.hidden_dim : j * self.hidden_dim + self.hidden_dim] = T
      T = T.matmul(self.A)
  
    return M

  @utils.profile
  def forward(self, step, x, y):

    # construct cholesky(MM') asynchronously with frame encoding
    # TODO: confirm that this stream stuff is working/helping
    m_stream = torch.cuda.Stream()
    with torch.cuda.stream(m_stream):
      M = self.build_M(sequence_length=x.size(1))
      MT = M.t()
      MMT = M.matmul(MT)
      U = torch.potrf(MMT, upper=False)

    # fold sequence into batch
    xflat = x.reshape(-1, x.size(2), x.size(3), x.size(4))
    
    # encode each frame of each sequence in parallel
    loss, (zcat, z, loglikelihood_accum) = self.flow(step, xflat)
    
    # unfold batch back into sequences
    # y is (batch, sequence, observation_dim)
    y = zcat.reshape(x.size(0), x.size(1), -1) 

    # stack sequence into tall vector
    # Y is tall (batch, sequence * observation_dim) with stacked y's
    Y = y.reshape(y.size(0), -1)

    m_stream.synchronize()
    z = torch.trtrs(Y.t(), U, transpose=False, upper=False)[0]
    z = torch.trtrs(z, U, transpose=True, upper=False)[0]
    xe = MT.matmul(z).t()
    
    # minimize || E ||
    epsilon = xe[:, self.hidden_dim:]
    prediction_loss = (epsilon * epsilon).mean() + loss

    return prediction_loss, None

  @utils.profile
  def logger(self, step, data, out):
    # TODO: log prediction and ground truth sequences
    return None #self.flow.logger(step, data, out)


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

