import libs.args.args as args
from libs.args.args import argchoice

import torch
import models
import datasets
import trainer

import inspect
import libs.colors.colors as colors
import utils

args.module('model', models.ReversibleFlow)
args.module('optimizer', torch.optim, default=torch.optim.Adam)
args.module('train_dataset', datasets)
args.module('validation_dataset', datasets)
args.module('trainer', trainer.Trainer)

args.arguments(epochs=1, name='', batch_size=32, resume=False, resume_uid='', resume_tag='best', log_frequency=10, 
               validation_frequency=5000, checkpoint_frequency=1000, cuda=True, print_model=False, max_grad=15)

args.defaults({'optimizer.lr': 1e-4})

pargs = args.reader()

train_dataset = pargs.train_dataset()
validation_dataset = pargs.validation_dataset()
model = pargs.model(train_dataset)
optimizer = pargs.optimizer(model.parameters())
trainer = pargs.trainer(logname=pargs.stub())

if pargs.cuda:
  model.cuda()
  trainer.cuda()

if pargs.resume:
  with utils.block('Resume', exit_on_error=True) as b:
    path = trainer.resume(model, pargs.resume_tag, uid=pargs.resume_uid, unique=False)
    b.print('Successfully Resumed: {}'.format(path))
    b.print(trainer.state_dict)
    
with utils.block('Command') as b:
  b.print(pargs.command())
  b.print('Stub: ' + pargs.stub())
  trainer.log(0, command=pargs.command())

with utils.block('Arguments') as b:
  b.print(pargs, indent=False)
  trainer.log(0, arguments=repr(pargs))

with utils.block('Model') as b:
  b.print(utils.pytorch_summary(model, verbose=pargs.print_model), indent=False)

@utils.profile
def main():
  for (epoch, batch, steps, bar), data in trainer(train_dataset, epochs=pargs.epochs, progress='Training', shuffle=True, batch_size=pargs.batch_size):
    optimizer.zero_grad()
    model.train() if not model.training else None
    
    loss_value, out = model(*data)
    loss_value.backward()

    if (steps + 1) % pargs.log_frequency == 0:
      trainer.log(steps,
                  loss=loss_value,
                  m=model.log(loss=loss_value, 
                                  out=out,
                                  data=data,
                                  epoch=epoch,
                                  batch=batch,
                                  step=steps))

    torch.nn.utils.clip_grad_value_(model.parameters(), pargs.max_grad)
    optimizer.step()

    if (steps + 1) % pargs.validation_frequency == 0:
      model.eval()
      validation_loss = sum(model(*data)[0].item() for _, data in trainer(validation_dataset, batch_size=pargs.batch_size, progress='Validation', grad=False, leave=False)) 
      validation_loss /= len(validation_dataset)
      trainer.log(steps, validation=validation_loss)

      if validation_loss > trainer.state_dict.get('best_validation_loss', 0):
        trainer.state_dict['best_validation_loss'] = validation_loss
        path = trainer.checkpoint(model, 'best')
        bar.write('Checkpoint at {}: {}'.format(steps, path))

    if (steps + 1) % pargs.checkpoint_frequency == 0:
      path = trainer.checkpoint(model, 'recent')
      bar.write('Checkpoint at {}: {}'.format(steps, path))

main()

# TODO: dynamic system dataset
# TODO: dockerfile / DGX tests
# TODO: multi-gpu integration for Trainer
# TODO: handle "data:type=None" for args
# TODO: complex pointwise multiply / dynamic system inference
# TODO: args README
# TODO: args @argignore and @arginclude
# TODO: experiment with normal_() again
# TODO: experiment with a single dimension of data, the rest filled with zeros (low rank data)
# TODO: why does negative scaling not seem to work as well?

print()