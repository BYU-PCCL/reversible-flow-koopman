
class Default(object):
  def __init__(self, optimizer):
    self.__dict__.update(locals())

  def step(self, step):
    return step == 0

  def lr(self, step):
    for param_group in self.optimizer.param_groups:
      return param_group['lr']


class TwoStage(object):
  def __init__(self, optimizer, threshold=10000, before=1e-4, after=1e-3):
      self.__dict__.update(locals())

  def step(self, step):
    lr = self.lr(step)
    
    response = False
    for param_group in self.optimizer.param_groups:
      response = (param_group['lr']  != lr) or response
      param_group['lr'] = lr

    return response or step == 0

  def lr(self, step):
      return self.before if step < self.threshold else self.after

