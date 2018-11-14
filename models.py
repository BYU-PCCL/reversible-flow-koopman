import torch
import torch.nn as nn
import torch.nn.functional as F
from libs.args.args import argchoice, argignore
import numpy as np
import pdb
import utils

from modules import ReversibleFlow, Network

class EncodeStateDirectly(nn.Module):
  def __init__(self, dataset, extra_hidden_dim=20, flow:argchoice=[ReversibleFlow]):
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

  @utils.profile
  def forward(self, step, x, y):

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

    # construct M
    sequence_length = y.size(1)
    M = torch.zeros((sequence_length * self.observation_dim, sequence_length * self.hidden_dim), device=self.A.device)
    T = self.C
    for offset in range(sequence_length):
      for x, y, in zip(range(sequence_length - offset), range(sequence_length - offset)):
        x = x + offset
        M[x * self.observation_dim:x * self.observation_dim + self.observation_dim, y * self.hidden_dim : y * self.hidden_dim + self.hidden_dim] = T
      T = T.matmul(self.A)
    
    # solve ME = Y for E
    Mdagger =  torch.pinverse(M).unsqueeze(0)
    xe = Mdagger.matmul(Y.unsqueeze(2))

    # This isn't correct... yet
    #U = torch.potrf(M.matmul(M.t()), upper=True)
    #Mi = M.t().matmul(torch.potri(U)).unsqueeze(0)
    #xei = Mi.matmul(Y.unsqueeze(2))

    #print(M.t().matmul(Mi).unsqueeze(0) - Mdagger)
    #print(Y.size(), M.size())
    #xechol = torch.potrs(Y, U, upper=True)
    #print(xechol.matmul(M).unsqueeze(2) - xe)
    
    # minimize || E || 
    prediction_loss = (xe * xe).mean()

    return prediction_loss, None

  @utils.profile
  def logger(self, step, data, out):
    return None #self.flow.logger(step, data, out)


# get batch of sequences
# encode each frame in sequence as y

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

