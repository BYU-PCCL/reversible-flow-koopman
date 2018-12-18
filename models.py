import torch
import torch.nn as nn
import torch.nn.functional as F
from libs.args.args import argchoice, argignore
import numpy as np
import pdb
import utils

from modules import ReversibleFlow, Network, AffineFlowStep

from hessian import jacobian

class Unitary(nn.Module):
  def __init__(self, dim):
    self.__dict__.update(locals())
    super(Unitary, self).__init__()
    assert ((dim & (dim - 1)) == 0) and dim > 0, 'dim must be a power of two'
    self.U = nn.Parameter(torch.Tensor(dim, dim))
    self.reset_parameters()
  
  def reset_parameters(self):
    torch.nn.init.xavier_uniform_(self.U)
    self.U.data /= self.U.norm(dim=1, keepdim=True)

  # def multi_matmul(self, M, left=None, right=None):
  #   if right - left == 1:
  #     return M[left]

  #   return torch.matmul(self.multi_matmul(M, left=(left + right)//2, right=right), 
  #                       self.multi_matmul(M, left=left, right=(left + right)//2))

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
    U = self.multi_matmul_power_of_two(UH)

    return U


class FramePredictionBase(nn.Module):
  def __init__(self, dataset, extra_hidden_dim=8, l2_regularization=.001,
               future_sequence_length=50,
               max_hidden_dim=128,
               true_hidden_dim=32,
               network:argchoice=[AffineFlowStep],
               flow:argchoice=[ReversibleFlow], inner_minimization_as_constant=True):
    self.__dict__.update(locals())
    super(FramePredictionBase, self).__init__()
    
    example = torch.stack([dataset[i][0][0] for i in range(20)])

    self.flow = flow(examples=example)    
    self.observation_dim = torch.numel(example[0])
    self.example = example[0].unsqueeze(0)
    
    # round up to the nearest power of two, makes multi_matmul significantly faster
    self.true_hidden_dim = true_hidden_dim
    self.hidden_dim = min(self.observation_dim + extra_hidden_dim, max_hidden_dim)
    #self.hidden_dim = 1 << (self.hidden_dim - 1).bit_length()

    self.U = Unitary(dim=self.hidden_dim)
    self.Q = Unitary(dim=self.hidden_dim)

    self.V = Unitary(dim=self.hidden_dim)
    self.alpha = nn.Parameter(torch.Tensor(self.true_hidden_dim))
    self.C = nn.Parameter(torch.Tensor(self.observation_dim, self.hidden_dim))
    # self.A = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))

    self.reset_parameters()

  def __repr__(self):
    summary =  '                Name : {}\n'.format(self.__class__.__name__)
    summary += '    Hidden Dimension : {}\n'.format(self.hidden_dim)
    summary += 'Obervation Dimension : {}\n'.format(self.observation_dim)
    summary += '         A Dimension : {0}x{0} = {1}\n'.format(self.hidden_dim, self.hidden_dim**2)
    # summary += '         C Dimension : {0}x{1} = {2}\n'.format(self.observation_dim, 
    #                                                            self.hidden_dim,
    #                                                            self.C.numel())
    return summary

  def reset_parameters(self):
    # Initialize a full-rank C
    torch.nn.init.xavier_uniform_(self.C)
    u, s, v = torch.svd(self.C)
    self.C.data = u.matmul(v.t())

    # torch.nn.init.xavier_uniform_(self.A)
    # u, s, v = torch.svd(self.A)
    # self.A.data = u.matmul(v.t())

    self.U.reset_parameters()
    self.V.reset_parameters()

    # We want to initialize so that the singular values are near .95
    torch.nn.init.constant_(self.alpha, -5.38113)
    torch.nn.init.constant_(self.alpha, -2 * np.pi)
    torch.nn.init.constant_(self.alpha, 0.001)

  def SV(self):
    #-torch.cos(self.alpha / 2.) / 2 + .5
    return 1 - torch.clamp(torch.abs(self.alpha), min=0, max=1)

  @utils.profile
  def build_A(self):
    U = self.U.get()
    V = self.V.get()
    S = self.SV()
    A = U[:, :self.true_hidden_dim].matmul(S[:, None] * V[:self.true_hidden_dim])

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

    # S = -torch.cos(self.alpha / 2.) / 2 + .5

    # Atest = U.matmul(S[None, :] * Vt)

    # print(A - Atest)
    # exit()

    A.retain_grad()
    return A
  
  '''
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
  '''

  @utils.profile
  def build_O_A(self, y, sequence_length, device, step):
    A, C = self.build_A(), self.C
    #A, C = self.n4sid(y)

    #if not hasattr(self, 'A'):
    # A, C = utils.N4SID_simple(y[0].detach().cpu().numpy().T, 2, y.size(1) // 2, self.hidden_dim)
    # A = torch.from_numpy(A).float().to(y.device)
    # C = torch.from_numpy(C).float().to(y.device)

    # np.save('/mnt/pccfs/backed_up/robert/A.npy', A.detach().cpu().numpy())
    # np.save('/mnt/pccfs/backed_up/robert/C.npy', C.detach().cpu().numpy())
    # np.save('/mnt/pccfs/backed_up/robert/y.npy', y.detach().cpu().numpy())

    # self.A = A.detach()
    # self.C = C.detach()

      # np.save('/mnt/pccfs/backed_up/robert/A.npy', A.detach().cpu().numpy())
      # np.save('/mnt/pccfs/backed_up/robert/C.npy', C.detach().cpu().numpy())

    # else:
    #   A = self.A
    #   C = self.C

    # C = C.detach()

    # print('mean, absmean, max, min')
    # print(C.mean().item(), C.abs().mean().item(), C.max().item(), C.min().item())

    O = [C]
    for t in range(sequence_length - 1):
       O.append(O[-1].matmul(A))
    O = torch.cat(O, dim=0)

    return O, A, C

  def n4sid(self, ys):

    #ys = ys.norm(dim=2, keepdim=True)
    
    b, t, o = ys.size()
    
    assert t - self.hidden_dim + 1 > 0, 'need more sequence frames than hidden dimensions'

    G = torch.as_strided(ys, (b, t - self.hidden_dim + 1, o, self.hidden_dim), 
                             (t * o, o, 1, o)).reshape(b, -1, self.hidden_dim).clone()

    # assert G.size(0) == 1, 'needs to be tested for batch > 1'
    # U, S, V = torch.svd(G[0])
    # G = U[None]

    G1 = G[:, :-o].reshape(-1, self.hidden_dim)
    G2 = G[:, o:].reshape(-1, self.hidden_dim)
    
    Ch = G[:, :o].mean(0)
    out, _ = torch.gels(G2, G1)
    Ah = out[:G1.size(1)]
    
    return Ah.detach(), Ch.detach()


  @utils.profile
  def forward(self, step, x, y):
    torch.set_printoptions(edgeitems=10, linewidth=200, precision=4)

    # print('-------------------')

    # for o in gc.get_objects():
    #   if torch.is_tensor(o):
    #     print(o.size())
    # print('-------------------')

    if step % 10 == 0:
      self.eval()

    x.requires_grad = True

    # fold sequence into batch
    xflat = x.reshape(-1, x.size(2), x.size(3), x.size(4))
    
    # encode each frame of each sequence in parallel
    glowloss, (zcat, zlist, logdet) = self.flow(step, xflat)

    # unfold batch back into sequences
    # y is (batch, sequence, observation_dim)
    y = zcat.reshape(x.size(0), x.size(1), -1)

    #mu = (y.reshape(-1, y.size(-1)).abs()).mean(dim=0)
    #y /= mu # all the y-dimensions are have mean abs of 1

    # stack sequence into tall vector
    # Y is tall (batch, sequence * observation_dim) with stacked y's
    Y = y.reshape(y.size(0), -1)

    # Find x0*

    M = float(np.prod(xflat.shape[1:]))

    if True:
      O, A, C = self.build_O_A(y=y, sequence_length=x.size(1), device=x.device, step=step)

      if self.inner_minimization_as_constant:
        # very fast, uses QR decomposition, not differentiable
        #if not hasattr(self, 'x0'):
        z, _ = torch.gels(Y.t(), O)
        x0 = z[:O.size(1)].detach()
        #self.x0 = x0
        #np.save('/mnt/pccfs/backed_up/robert/x0.npy', x0.detach().cpu().numpy())

        # else:
        #   x0 = self.x0

        #print('resids on x0', z[O.size(1):])

        # for rank deficent case
        #x0, resid, rank, sv = np.linalg.lstsq(O.detach().cpu().numpy(), Y.t().detach().cpu().numpy(), rcond=1e-3)
        #x0 = torch.from_numpy(x0).to(O.device)
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
      
      # Yhat = Yhat.view(y.size())
      # for i in range(Yhat.size(1)):
      #   Yhat[:, i] *= 0
      #   Yhat[:, i] += 1
        
      #Yhat = Yhat.view(y.size(0), -1)


      # print('-------------------')

      # for o in gc.get_objects():
      #   if torch.is_tensor(o):
      #     print(o.size())
      # print('-------------------')

      


      # print(torch.cuda.memory_allocated())

      w = Y.t() - Yhat
      prediction_error = (w.t().reshape(y.size()).norm(dim=2)**2).mean()  # np.log(2 * np.pi)

    else:
      w = None
      prediction_error = None
      O = None
      C = None
      A = None
      x0 = None
      Yhat = None

    # if not hasattr(self, 'yhat'):
    #   self.yhat = Y.t().detach().clone()
    #   self.yhat.data.normal_()
    #   Yhat = self.yhat
    
    # Ytarget = self.yhat

    #w = Y.t() - Yhat(Ytarget + torch.rand_like(Y.t()) * 0.1)
    #logdet_error = ((logdet / M)**2).mean()

    # prediction_error

    # glowloss + .5 *

    loss =  .5 * (w * w).mean() - (logdet / M).mean()

    #Yhat = Y.t() + torch.rand_like(Y.t()) * .01

    # increase variance - doesn't help
    # add noise to y - it just gets removed during the backward pass.. it helps logging.. but not optimization
    # train dynamical system on raw data
    # n4sid in-the-loop
    # additive glow only

    # Yhat = Y.t() + torch.randn_like(Y.t()) * .3

    return loss, (loss, w, Y, Yhat, y, A, x0, logdet / M, prediction_error, zlist, zcat, O, C)

  def decode(self, Y, sequence_length, zlist):
    # heaven help me if I need to change how this indexing works
    # this is awkward, but we are converting shapes so that we go from
    # Yhat to Y to y to zcat to zlist then decoding and reshaping to compare with x
    s = sequence_length
    n = Y.size(0)

    zcat_hat = Y.reshape(-1, self.observation_dim)
    indexes = np.cumsum([0] + [m[:n * s].view(n * s, -1).size(1) for m in zlist])
    zlist_hat = [zcat_hat[:, a:b].reshape(c[:n * s].size()) for a, b, c in zip(indexes[:-1], indexes[1:], zlist)]
    xhat, _ = self.flow.decode(zlist_hat)

    return xhat

  @utils.profile
  def logger(self, step, data, out):

    # print([n for n, p in self.named_parameters() if p.grad is not None])
    # print([(n, p.grad.norm().item()) for n, p in self.named_parameters() if p.grad is not None])
    self.eval()

    x, yin = data
    b, s, c, h, w = x.shape
    loss, w, Y, Yhat, y, A, x0, logdet, prediction_error, zlist, zcat, O, C = out
    stats = {':logdet': logdet.mean(),
             ':predmean': w.mean(),
             ':predstd': w.std(),
             #':Atheta': torch.arccos(.5 * (torch.trace(A) - 1)) # still untested: https://math.stackexchange.com/questions/261617/find-the-rotation-axis-and-angle-of-a-matrix
             }

    errnorm = w.t().reshape(y.size()).norm(dim=2)

    # print([n for n, p in self.named_parameters() if p.grad is not None])
    # print([(n, p.grad.norm().item()) for n, p in self.named_parameters() if p.grad is not None])

    # xdiff = (x[:, 1:] - x[:, :-1]).reshape(y.size(0), -1, y.size(2))
    # ydiff = y[:, 1:] - y[:, :-1]

    stats.update({'pe:prediction_error': prediction_error,
                  ':log10_prediction_error': torch.log(prediction_error),
                  'logdet:loge_det': logdet.mean(),
                  ':y_mean': y.mean(),
                  ':y_max': y.max(),
                  ':y_min': y.min(),
                  ':log10_first_errnorm': torch.log(errnorm[:, 0].mean()),
                  ':log10_last_errnorm': torch.log(errnorm[:, -1].mean()),
                  ':last_over_first_errnorm': (errnorm[:, -1] / errnorm[:, 0]).mean()
                  })

    # stats.update({'|y|:mean_y_norm': y.norm(dim=2).mean(),
    #               '|x|:mean_x0_norm': x0.norm(dim=0).mean(),
    #               '|x|:mean_x_norm': x.reshape(x.size(0), x.size(1), -1).norm(dim=2).mean(),
    #               'xₘₐₓ:x_max':x0.max()
    #               })

    # stats.update({'gmax': max([p.grad.abs().max() for p in self.parameters() if p.grad is not None]),
    #               'gnorm': max([p.grad.abs().norm() for p in self.parameters() if p.grad is not None])
    #                })

    # # if step % 10**(int(np.log10(step + 99)) - 1) == 0:
    if step % 10 == 0:
      xhat = self.decode(Yhat.t()[:1], s, zlist)

      #render out into the future
      state = x0[:, 0:1]
      for i in range(s):
        state = A.matmul(state)
      yfuture = [C.matmul(state)]
      future_sequence_length = self.future_sequence_length
      for i in range(future_sequence_length - 1):
        state = A.matmul(state)
        yfuture.append(C.matmul(state))
      yfuture = torch.cat(yfuture, dim=0).t()
      xfuture = self.decode(yfuture, future_sequence_length, zlist)
      yfuture = yfuture.reshape(1, future_sequence_length, *x[0:1].shape[2:])

      # prepare the ys for logging
      recon_error = (x[0] - xhat) * self.dataset.normalizing_const
      ytruth =      Y[0:1].reshape(y[0:1].size()).reshape(x[0].size())
      yhat = Yhat.t()[0:1].reshape(y[0:1].size()).reshape(x[0].size())
      
      xerror = recon_error.abs().max(1)[0].detach()
      xerror = xerror.reshape(xerror.size(0), -1)
      xerror = xerror.clamp(min=0, max=1)
      xerror = xerror.reshape(recon_error.size(0), 1, recon_error.size(2), recon_error.size(3))

      stats.update({#'recon': recon_error.abs().mean(),
                    ':yhat': yhat,
                    ':ytruth': ytruth, 
                    ':yerr': (yhat - ytruth),
                    #':yerror': (yhat - ytruth).view(-1),
                    ':reconstruction_error': recon_error.view(-1),
                    ':xerror': xerror,
                    ':xhat': xhat * self.dataset.normalizing_const,
                    ':xfuture': xfuture * self.dataset.normalizing_const,
                    #':xtruth': x[0] * self.dataset.normalizing_const
                    })

    # if step % 100 ==  0:
    #   xflatsingle = x.reshape(-1, x.size(2), x.size(3), x.size(4))
    #   xflatsingle = xflatsingle[0:1].clone()
    #   xflatsingle.requires_grad = True
    #   _, (zcatsingle, zlistsingle, _) = self.flow(step, xflatsingle)
    #   jac = jacobian(zcatsingle, xflatsingle)
    #   svs = torch.svd(jac)[1]

    #   stats.update({':jacforward': jac.unsqueeze(0),
    #                 ':jacforward_norm_max': svs.max(),
    #                 ':jacforward_norm_min': svs.min(),
    #                 ':jacforward_svs': svs,
    #                 ':detjacforward': torch.det(jac)})

    #   xlistsingle, _ = self.flow.decode(zlistsingle)

    #   jac = jacobian(xlistsingle, zlistsingle)
    #   svs = torch.svd(jac)[1]
    #   stats.update({':jacbackward': jac.unsqueeze(0),
    #                 ':jacbackward_norm_max': svs.max(),
    #                 ':jacbackward_norm_min': svs.min(),
    #                 ':jacbackward_svs': svs,
    #                 ':detjacbackward': torch.det(jac)})

    svs = torch.svd(A)[1]

    stats.update({#':gO': O.grad.unsqueeze(0),
                  ':A': A.unsqueeze(0),
                  #':gA': A.grad.reshape(-1) if A.grad is not None else None,
                  ':C': C.unsqueeze(0),
                  #':gC': self.C.grad.reshape(-1) if self.C.grad is not None else None,
                  ':max_sv(A)': svs.max(),
                  ':min_sv(A)': svs.min(),
                  #':Ad':A.reshape(-1),
                  #':Csv': torch.svd(self.C)[1],
                  #':Cd':C.reshape(-1)
                  })


    self.train()

    return stats


# Learning at various time scales
# learning image deltas
# higher resolution
# PCA
# i still don't understand why long sequences suffer



# if the y's scale down by 10, the x0 scales down by 10, and the loss scales down by 10
# how can we (properly) be agnostic to the scale of y?

# log it/s?
# large state spaces struggle to learn?

# WHY DOES EXACT PARAMETERIZATION FAIL?
  # - one global minima?
  # - slow learning rate on matricies relative to encoder?
# How can we get perfect prediction loss, but non-zero reconstruction loss, with a logdet == 1?

#############################
#############################
# FACTOR INTO TESTABLE HYPTOTHESIS
#############################
#############################

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

