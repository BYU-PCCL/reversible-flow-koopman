import libs.args.args as args
from libs.args.args import argchoice

import torch
import models
import datasets
import schedules
import trainer

import inspect
import libs.colors.colors as colors
import utils
import atexit

#from apex import amp

args.module('model', [models.FramePredictionBase, models.GlowPrediction])
args.module('optimizer', torch.optim, default=torch.optim.Adam)
args.module('train_dataset', datasets)
args.module('validation_dataset', datasets)
args.module('trainer', trainer.Trainer)
args.module('scheduler', schedules)

args.arguments(epochs=100, name='exp', batch_size=100, resume='', resume_uid='', shuffle_training=True,
               validation_frequency=5000, checkpoint_frequency=1000, cuda=True, print_model=False, max_grad=3, 
               max_grad_norm=20, amp=True, checkpoint=True)

args.defaults({'optimizer.lr': .0001})

pargs = args.reader()

torch.manual_seed(40)

train_dataset = pargs.train_dataset()
# validation_dataset = pargs.validation_dataset(train=False)
model = pargs.model(train_dataset)
model_obj = model


optimizer = pargs.optimizer(model.parameters())
optimizer = pargs.optimizer([{'params': model.flow.parameters()},
                             {'params': model.parameters(recurse=False), 'lr': 0.01}] +
                             ([{'params': model.A.parameters(), 'lr': 0.01}] if hasattr(model, 'A') else []))
trainer = pargs.trainer(_logname=pargs.name)
scheduler = pargs.scheduler(optimizer)

if pargs.cuda:
 #model = torch.nn.DataParallel(model)
  #model_obj = model.module
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
  trainer.log(0, **{':command': pargs.command()})

with utils.block('Arguments') as b:
  b.print(pargs, indent=False)
  trainer.log(0, **{':arguments': repr(pargs)})

with utils.block('Parameters') as b:
  b.print(utils.pytorch_summary(model, verbose=pargs.print_model), indent=False)

with utils.block('Model') as b:
  b.print(repr(model_obj))

with utils.block('Train Dataset') as b:
  b.print(repr(train_dataset))

@utils.profile
def main():

  #amp_handle = amp.init(enabled=True)

  # save a checkpoint at exit, regardless of the reason
  if pargs.checkpoint:
    atexit.register(trainer.checkpoint, model, optimizer, tag='exit', log=print)

  for (epoch, batch, step, bar), data in trainer(train_dataset, epochs=pargs.epochs, 
    progress='Training', shuffle=pargs.shuffle_training, batch_size=pargs.batch_size):

    trainer.state_dict['step'] = step + 1 # resume starting on the next step

    optimizer.zero_grad()

    model.train() if not model.training else None

    loss_value, out = model(step, *data)
    loss_value = loss_value.mean() # multi-gpu accumulation
    torch.cuda.synchronize()

    loss_value.backward()
    torch.cuda.synchronize()

    trainer.log(step, loss=loss_value, m=model_obj.logger(step, data, out))
    
    #with amp_handle.scale_loss(loss_value, optimizer) as scaled_loss:
    #  scaled_loss.backward()
    
    # TODO: validation logging isn't working well, but I don't need it right now
    # if step % pargs.validation_frequency == 0 and step != 0:
    #   model.eval()
    #   val_gen = trainer(validation_dataset, batch_size=pargs.batch_size, progress='Validation', grad=False, leave=False)
    #   validation_loss = sum(model(*data)[0].item() for _, data in val_gen) / len(val_gen)
    #   trainer.log(step, validation=validation_loss)

    #   if validation_loss < trainer.state_dict.get('best_validation_loss', validation_loss + 1):
    #     trainer.state_dict['best_validation_loss'] = validation_loss
    #     path = trainer.checkpoint(model, optimizer, tag='best_validation', log=bar.write)

    # if step % pargs.checkpoint_frequency == 0  and step != 0:
    #   path = trainer.checkpoint(model, optimizer, tag='recent', log=bar.write)

    #   trainer.state_dict['running_training_loss'] = .95 * trainer.state_dict.get('running_training_loss', loss_value) + .05 * loss_value
    #   if trainer.state_dict['running_training_loss'] < trainer.state_dict.get('best_training_loss', 1e10):
    #     trainer.state_dict['best_training_loss'] = trainer.state_dict['running_training_loss']
    #     path = trainer.checkpoint(model, optimizer, tag='best_training', log=bar.write)

    #torch.nn.utils.clip_grad_value_(model.parameters(), pargs.max_grad)
    torch.nn.utils.clip_grad_norm_(model.parameters(), pargs.max_grad_norm)
    optimizer.step()

    if scheduler.step(step):
      trainer.log(step, lr=scheduler.lr(step))

    del loss_value
    del out

main()

# TODO: multi-gpu integration for Trainer
# TODO: complex pointwise multiply / dynamic system inference

# TODO: handle "data:type=None" for args
# TODO: args README
# TODO: args @argignore and @arginclude
# TODO: boolean args with default=False should be "store_true" in addition to --arg=True

# TODO: experiment with a single dimension of data, the rest filled with zeros (low rank data)
# TODO: why does negative scaling not seem to work as well?


# fft?
# rewrite checkpointing api


print()