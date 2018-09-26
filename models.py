import torch
import torch.nn as nn
from libs.args.args import argchoice

class Model(nn.Module):
  def __init__(self, num_layers=5, loss:argchoice=[nn.MSELoss]):
  	super(Model, self).__init__()
  	self.a = nn.Parameter(torch.zeros(1))
  	self.loss = loss()

  def forward(self, x, y):
  	yhat = y.float() * self.a
  	loss = self.loss(yhat, y.float())
  	return loss, yhat

  def log(self, **kwargs):
  	return {}