import libs.args.args as args
from libs.args.args import argchoice

import torch
import models
import datasets
import trainer

import inspect
import libs.colors.colors as colors
import utils
import atexit

args.module('model', models.ReversibleFlow)
args.module('optimizer', torch.optim, default=torch.optim.Adam)
args.module('train_dataset', datasets)
args.module('validation_dataset', datasets)
args.module('trainer', trainer.Trainer)

args.arguments(epochs=1, name='', batch_size=32, resume='', resume_uid='', log_frequency=10, 
               validation_frequency=5000, checkpoint_frequency=1000, cuda=True, print_model=False, max_grad=40)

args.defaults({'optimizer.lr': 1e-3})

pargs = args.reader()

train_dataset = pargs.train_dataset()
validation_dataset = pargs.validation_dataset(train=False)
model = pargs.model(train_dataset)
optimizer = pargs.optimizer(model.parameters())
trainer = pargs.trainer(_logname=pargs.stub())

if pargs.cuda:
  model.cuda()
  trainer.cuda()

if len(pargs.resume) > 0:
  with utils.block('Resume', exit_on_error=True) as b:
    path = trainer.resume(model, optimizer, pargs.resume, uid=pargs.resume_uid, unique=False)
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

  # save a checkpoint at exit, regardless of the reason
  atexit.register(trainer.checkpoint, model, optimizer, tag='exit', log=print)

  for (epoch, batch, steps, bar), data in trainer(train_dataset, epochs=pargs.epochs, progress='Training', shuffle=True, batch_size=pargs.batch_size):
    
    trainer.state_dict['step'] = steps + 1 # resume starting on the next step

    optimizer.zero_grad()
    model.train() if not model.training else None
    
    loss_value, out = model(*data)
    loss_value.backward()

    if steps % pargs.log_frequency == 0:
      trainer.log(steps,
                  loss=loss_value,
                  m=model.log(loss=loss_value, 
                              out=out,
                              data=data,
                              epoch=epoch,
                              batch=batch,
                              step=steps))
    
    if steps % pargs.validation_frequency == 0 and steps != 0:
      model.eval()
      val_gen = trainer(validation_dataset, batch_size=pargs.batch_size, progress='Validation', grad=False, leave=False)
      validation_loss = sum(model(*data)[0].item() for _, data in val_gen) / len(val_gen)
      trainer.log(steps, validation=validation_loss)

      if validation_loss < trainer.state_dict.get('best_validation_loss', validation_loss + 1):
        trainer.state_dict['best_validation_loss'] = validation_loss
        path = trainer.checkpoint(model, optimizer, tag='best_validation', log=bar.write)

    if steps % pargs.checkpoint_frequency == 0  and steps != 0:
      path = trainer.checkpoint(model, optimizer, tag='recent', log=bar.write)

      trainer.state_dict['running_training_loss'] = .95 * trainer.state_dict.get('running_training_loss', loss_value) + .05 * loss_value
      if trainer.state_dict['running_training_loss'] < trainer.state_dict.get('best_training_loss', 1e10):
        trainer.state_dict['best_training_loss'] = trainer.state_dict['running_training_loss']
        path = trainer.checkpoint(model, optimizer, tag='best_training', log=bar.write)

    torch.nn.utils.clip_grad_norm_(model.parameters(), pargs.max_grad)
    optimizer.step()

main()

# TODO: dockerfile / DGX tests
# TODO: multi-gpu integration for Trainer
# TODO: dynamic system dataset
# TODO: complex pointwise multiply / dynamic system inference

# TODO: handle "data:type=None" for args
# TODO: args README
# TODO: args @argignore and @arginclude

# TODO: experiment with a single dimension of data, the rest filled with zeros (low rank data)
# TODO: why does negative scaling not seem to work as well?

print()