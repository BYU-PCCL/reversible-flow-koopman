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

    self.U.data *= 0
    self.U.data.view(-1)[::self.U.size(1) + 1] += 1

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
               future_sequence_length=5,
               max_hidden_dim=128,
               log_all=True,
               prediction_alpha=1,
               #true_hidden_dim=32,
               network:argchoice=[AffineFlowStep],
               flow:argchoice=[ReversibleFlow], inner_minimization_as_constant=True):
    self.__dict__.update(locals())
    super(FramePredictionBase, self).__init__()
    
    example = torch.stack([dataset[i][0][0] for i in range(20)])

    self.flow = flow(examples=example)    
    self.observation_dim = torch.numel(example[0])
    self.example = example[0].unsqueeze(0)
    
    # round up to the nearest power of two, makes multi_matmul significantly faster
    self.hidden_dim = min(self.observation_dim + extra_hidden_dim, max_hidden_dim)
    #self.hidden_dim = 1 << (self.hidden_dim - 1).bit_length()
    #self.true_hidden_dim = min(true_hidden_dim, self.hidden_dim)

    self.U = Unitary(dim=self.hidden_dim)
    self.V = Unitary(dim=self.hidden_dim)
    self.alpha = nn.Parameter(torch.Tensor(self.hidden_dim))
    self.C = nn.Parameter(torch.Tensor(self.observation_dim, self.hidden_dim))
    # self.A = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))

    self.eig_alpha = nn.Parameter(torch.Tensor(self.hidden_dim // 2).double())
    self.eig_beta = nn.Parameter(torch.Tensor(self.hidden_dim // 2).double())

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
    # torch.nn.init.constant_(self.alpha, -5.38113)
    # torch.nn.init.constant_(self.alpha, -2 * np.pi)
    torch.nn.init.constant_(self.alpha, 0.001)
    torch.nn.init.constant_(self.eig_alpha, 0.5)
    torch.nn.init.constant_(self.eig_beta, 0.5)

  def SV(self):
    #-torch.cos(self.alpha / 2.) / 2 + .5
    return 1 - torch.clamp(torch.abs(self.alpha), min=0, max=1)

  @utils.profile
  def build_A(self):
    # U = self.U.get()#.detach()
    # V = self.V.get()#.detach()
    # S = self.SV()
    # A = U[:, :self.hidden_dim].matmul(S[:, None] * V[:self.hidden_dim])

    # Real "Jordan" Form
    # To avoid sqrt(0) when alpha = 1, we use doubles for eig_alpha and eig_beta, and a very small
    # epsilon, it is cast back to a 32-bit float when the assignment happens to build the A matrix
    alpha = torch.clamp((1 - 1e-14) - torch.abs(self.eig_alpha), min=0)
    beta = torch.clamp(1 - torch.abs(self.eig_beta), min=0) * torch.sqrt(1 - alpha**2)

    A = torch.zeros(self.hidden_dim, self.hidden_dim).to(alpha.device)
    for i in range(self.hidden_dim // 2):
      A[2 * i, 2 * i] = alpha[i]
      A[2 * i + 1, 2 * i + 1] = alpha[i]
      A[2 * i, 2 * i + 1] = beta[i]
      A[2 * i + 1, 2 * i] = -beta[i]



    # print(np.trace(U.detach().cpu().numpy()))
    # print(torch.det(U))
    # print('trace', .5 * (np.trace(U.detach().cpu().numpy()) - 1))
    # print('U', np.arccos(.5 * (np.trace(U.detach().cpu().numpy()) - 1)))
    # print('V', np.arccos(.5 * (np.trace(V.detach().cpu().numpy()) - 1)))
    # print('A', np.arccos(.5 * (np.trace(Ar.detach().cpu().numpy()) - 1)))

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

    # A = A.detach()
    # C = C.detach()

    #if not hasattr(self, 'A'):
    # A, C = utils.N4SID_simple(y[0].detach().cpu().numpy().T, 2, y.size(1) // 2, self.hidden_dim)
    # A = torch.from_numpy(A).float().to(y.device)
    # C = torch.from_numpy(C).float().to(y.device)

    # np.save('/mnt/pccfs/backed_up/robert/A.npy', A.detach().cpu().numpy())
    # np.save('/mnt/pccfs/backed_up/robert/C.npy', C.detach().cpu().numpy())
    # np.save('/mnt/pccfs/backed_up/robert/y.npy', y.detach().cpu().numpy())

    #np.set_printoptions(suppress=True, precision=3)
    #print(1 / (np.linalg.eig(A.detach().cpu().numpy())[0].imag / (2 * np.pi)))

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

    # Normalize C
    # C = C / (C.max() - C.min())

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
    x.requires_grad = True


    # fold sequence into batch
    xflat = x.reshape(-1, x.size(2), x.size(3), x.size(4))
    M = float(np.prod(xflat.shape[1:]))

    #print('thing', xflat.mean(), xflat.var())
    
    # print('--')
    # print(xflat.view(xflat.size(0), -1)[0])

    # encode each frame of each sequence in parallel
    glowloss, (zcat, zlist, logdet) = self.flow(step, xflat)

    # unfold batch back into sequences
    # y is (batch, sequence, observation_dim)
    y = zcat.reshape(x.size(0), x.size(1), -1)#.detach()

    #mu = (y.reshape(-1, y.size(-1)).abs()).mean(dim=0)
    #y /= mu # all the y-dimensions are have mean abs of 1

    #print(zcat.size(), xflat.size())
    #print(xflat.view(xflat.size(0), -1)[0])
    
    #print((zcat / xflat.view(xflat.size(0), -1)))
    
    # stack sequence into tall vector
    # Y is tall (batch, sequence * observation_dim) with stacked y's
    Y = y.reshape(y.size(0), -1)

    # the scaling factor should not be centered (i.e not std or var, which are computed using centered values)
    scaling_lambda = y.abs().mean()

    # Find x0*
    O, A, C = self.build_O_A(y=y, sequence_length=x.size(1), device=x.device, step=step)

    if self.inner_minimization_as_constant or False:
      # very fast, uses QR decomposition, not differentiable
      #if not hasattr(self, 'x0'):
      # / (Y.max() - Y.min())
      z, _ = torch.gels(Y.t(), O)
      x0 = z[:O.size(1)].detach()

      # RO = O.clone()
      # RY = Y.t().clone()
      # R = 200
      # RY[R * 1024:] *= 0
      # RO[R * 1024:] *= 0

      # z, _ = torch.gels(RY, RO)
      # Rx0 = z[:O.size(1)].detach()

      # print(x0)
      # print(Rx0)
      # print(Rx0 - x0)
      # exit()

      #self.x0 = x0
      # np.save('/mnt/pccfs/backed_up/robert/x0.npy', x0.detach().cpu().numpy())

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
    
    Yhat = O.matmul(x0) # * 0 #+ 10

    # Yhat = Yha`t - Yhat.mean()
    # Yhat = Yhat / (Yhat.max() - Yhat.min())
    # Yhat = Yhat * 20

    #Yhat = Yhat / (Yhat.max() - Yhat.min()).detach()
    #Yhat = 20 * Yhat


    # print('yhat',  Yhat.min().item(), Yhat.max().item())
    # print('y',  Y.min().item(), Y.max().item())
    # print('scaling', Yhat.abs().mean())
    # print('xnorm', xflat.reshape(xflat.shape[0], -1).norm(dim=1).mean())

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

    # if step % 2 == 0:
    #   Yhat = Yhat.detach()
    # else:
    #   Y = Y.detach()
    #   logdet = logdet.detach()

    # this accounts for the scaling factor, but learns an unscaled dynamic system
    w = (Y.t() - Yhat) / scaling_lambda

    # print('w', w.std())
    # print('yhat', Yhat.abs().mean().item())
    # print('y', (Y.reshape(y.size()).std(dim=1)).mean())
    # print('scaling_lambda', scaling_lambda)

    #print('    x0:', x0.min().item(), x0.max().item(), x0.var().item(), x0.norm().item())
    # print('     w:', w.mean().item(), w.var().item(), w.max().item(), w.min().item())

    prediction_error = (w * w).mean()  # np.log(2 * np.pi)
    #prediction_error = ((w * w).reshape(y.size()) / y.norm(dim=2, keepdim=True).mean()).mean()

    xhat = self.decode(Yhat, x.size(1), zlist)
    reconstruction_loss = ((xhat - xflat)**2).mean()
    

    # if not hasattr(self, 'yhat'):
    #   self.yhat = Y.t().detach().clone()
    #   self.yhat.data.normal_()
    #   Yhat = self.yhat
    
    # Ytarget = self.yhat

    #w = Y.t() - Yhat(Ytarget + torch.rand_like(Y.t()) * 0.1)
    #logdet_error = ((logdet / M)**2).mean()

    # prediction_error

    # glowloss + .5 *

    # print((torch.exp(logdet / M)).mean())
    # print(x0)
    # print(x0.size())

    # print('************')
    # print('************')
    # print('************')
    # print('************')
    # print('************')
    # print('************')
    # print('************')
    # print('************')


    #loss = .5 * (w * w).mean() - (logdet / M).mean() 

    # prediction error by itself no longer has a bias to trend to small norm thanks to the normalization
    # what could cause the log determinant term to get smaller -- opportunity for better prediction error
    # what could prevent the log determinant term from getting bigger 
      # -- the gradients of prediction error could be larger 
    
    # print('net   part', (logdet / M).mean())
    # print('scale part',  torch.log(1. / scaling_lambda))

      # .999965 *
    log_likelihood = (logdet / M).mean() + torch.log(1. / scaling_lambda)
    loss = -log_likelihood + self.prediction_alpha * prediction_error #+ reconstruction_loss

    if np.isnan(loss.detach().cpu().numpy()):
      print('loss is nan')
      import pdb
      pdb.set_trace()

    # .5 * 1/err_std * prediction_error



    # print('max', (w**2).max())
    # print((logdet / M).max())

    # if step % 10 == 0:
    #   gpred = torch.autograd.grad(.5 * prediction_error, self.flow.parameters(), retain_graph=True)
    #   glogdet = torch.autograd.grad((logdet / M).mean(), self.flow.parameters(), retain_graph=True)

    #   for (n,v), gpred, glogdet in zip(self.flow.named_parameters(), gpred, glogdet):
    #     if 'bias' not in n and '4.weight' in n:
    #       print('{:>30} pred/logdet {:.4f} {:.4f} {:.4f}'.format(n, (gpred.norm() / glogdet.norm()).item(), gpred.max(), glogdet.max()))

    # print('--')
    # print(C.mean(), C.max(), C.min())
    # print('********************robert') 
    # print('pred', .5 * (w * w).mean())
    # print('logdet', (logdet / M).mean())
    # print(loss)
    # print('--')
    #loss *= 1 / x.size(1)

    #print(w.std())

    #Yhat = Y.t() + torch.rand_like(Y.t()) * 1

    # increase variance - doesn't help
    # add noise to y - it just gets removed during the backward pass.. it helps logging.. but not optimization
    # train dynamical system on raw data
    # n4sid in-the-loop
    # additive glow only

    # Yhat = Y.t() + torch.randn_like(Y.t()) * .3

    return loss, (loss, w, Y, Yhat, y, A, x0, logdet / M, prediction_error, zlist, zcat, O, C, scaling_lambda)

  def decode(self, Y, sequence_length, zlist):
    # heaven help me if I need to change how this indexing works
    # this is awkward, but we are converting shapes so that we go from
    # Yhat to Y to y to zcat to zlist then decoding and reshaping to compare with x
    s = sequence_length
    n = Y.size(0)

    zcat_hat = Y.reshape(-1, self.observation_dim)
    # print([m[:n * s].view(n * s, -1).size(1) for m in zlist])
    indexes = np.cumsum([0] + [np.prod(m.shape[1:]) for m in zlist])
    zlist_hat = [zcat_hat[:, a:b].reshape(c[:n * s].size()) for a, b, c in zip(indexes[:-1], indexes[1:], zlist)]
    xhat, _ = self.flow.decode(zlist_hat)

    return xhat

  @utils.profile
  def logger(self, step, data, out):

    # print([n for n, p in self.named_parameters() if p.grad is not None])
    # for n, v in [(n, p.grad) for n, p in self.named_parameters() if p.grad is not None]:
    #     if 'bias' not in n:
    #       print('{:>30} {:.3f} {:.3f}'.format(n, v.norm().item(), v.max().item()))
    #     if n == 'flow.flows.0.3.f.net.4.weight':
    #       print(v.norm(dim=1))

    #exit()
    self.eval()

    x, yin = data
    b, s, c, h, w = x.shape
    loss, w, Y, Yhat, y, A, x0, logdet, prediction_error, zlist, zcat, O, C, scaling_lambda = out
    stats = {':logdet': logdet.mean(),
             ':w_mean': w.mean(),
             ':w_std': w.std(),
             ':scaling_lambda': scaling_lambda,
             ':normed_prederr': ((w * w).reshape(y.size()) / y.norm(dim=2, keepdim=True)).mean()
             #':Atheta': torch.arccos(.5 * (torch.trace(A) - 1)) # still untested: https://math.stackexchange.com/questions/261617/find-the-rotation-axis-and-angle-of-a-matrix
             }

    errnorm = w.t().reshape(y.size()).norm(dim=2)

    # print(self.U.U.grad.norm())
    # print(self.V.U.grad.norm())
    # print(self.alpha.grad.norm())

    # for n, p in self.named_parameters():
    #   if p.grad is not None:
    #     if np.isnan(p.grad.max().detach().cpu().numpy()):
    #       print('{}: {}'.format(n, p.grad))
    #       import pdb
    #       pdb.set_trace()

    #  if p.grad is not None:
    #   print('{:>15}: {:.4f}'.format(n, p.grad.norm().item()))

    #print([(n, p.grad.norm().item()) for n, p in self.named_parameters() if p.grad is not None])

    # xdiff = (x[:, 1:] - x[:, :-1]).reshape(y.size(0), -1, y.size(2))
    # ydiff = y[:, 1:] - y[:, :-1]

    stats.update({'pe:prediction_error': prediction_error,
                  ':y_mean': y.mean(),
                  ':y_max': y.max(),
                  ':y_min': y.min(),
                  ':y_var': y.var(),
                  ':y_mean_norm': y.norm(dim=2).mean(),
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

    if self.log_all:
      if step % 100 == 0:
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
        recon_error = self.dataset.denormalize(x[0] - xhat)
        ytruth =      Y[0:1].reshape(y[0:1].size()).reshape(x[0].size())
        yhat = Yhat.t()[0:1].reshape(y[0:1].size()).reshape(x[0].size())
        
        xerror = recon_error.abs().max(1)[0].detach()
        xerror = xerror.reshape(xerror.size(0), -1)
        xerror = xerror.clamp(min=0, max=1)
        xerror = xerror.reshape(recon_error.size(0), 1, recon_error.size(2), recon_error.size(3))

        stats.update({#'recon': recon_error.abs().mean(),
                      ':yhat': (yhat - yhat.min()) / ((yhat - yhat.min())).max(),
                      ':ytruth': (ytruth - ytruth.min()) / ((ytruth - ytruth.min())).max(), 
                      ':yerr': (yhat - ytruth),
                      #':yerror': (yhat - ytruth).view(-1),
                      ':reconstruction_error': recon_error.view(-1),
                      ':xerror': xerror,
                      ':xhat': self.dataset.denormalize(xhat),
                      ':xfuture': self.dataset.denormalize(xfuture),
                      ':xtruth': self.dataset.denormalize(x[0])
                      })


      if step % 500 ==  0:
        xflatsingle = x.reshape(-1, x.size(2), x.size(3), x.size(4))
        xflatsingle = xflatsingle[0:1].clone()
        xflatsingle.requires_grad = True
        _, (zcatsingle, zlistsingle, _) = self.flow(step, xflatsingle)
        # jac = jacobian(zcatsingle, xflatsingle)
        # svs = torch.svd(jac)[1]

        # stats.update({':jacforward': jac.unsqueeze(0),
        #               ':jacforward_norm_max': svs.max(),
        #               ':jacforward_norm_min': svs.min(),
        #               ':jacforward_svs': svs})

        xlistsingle, _ = self.flow.decode(zlistsingle)

        jac = jacobian(xlistsingle, zlistsingle)
        svs = torch.svd(jac)[1]
        stats.update({':jacbackward': jac.unsqueeze(0),
                      ':jacbackward_norm_max': svs.max(),
                      ':jacbackward_norm_min': svs.min(),
                      ':jacbackward_percent_bad_svs': (svs > 1).float().mean()})

        u, svs, v = torch.svd(A)
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

