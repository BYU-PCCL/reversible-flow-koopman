import libs.args.args as args
from libs.args.args import argchoice

import torch
import models
import datasets
import trainer

import inspect
import libs.colors.colors as colors
import utils

args.module('model', models)
args.module('optimizer', torch.optim, default=torch.optim.Adam)
args.module('train_dataset', datasets)
args.module('validation_dataset', datasets)

args.arguments(epochs=1, batch_size=32, resume=True, log_frequency=1)

args.defaults({'dataset.subdataset.optimizer.lr': 1e-8})

pargs = args.reader()

train_dataset = pargs.train_dataset()
validation_dataset = pargs.validation_dataset()
model = pargs.model(train_dataset)
optimizer = pargs.optimizer(model.parameters())
trainer = trainer.Trainer()

with utils.block('Command Line') as b:
  b.print(pargs.command())

with utils.block('Arguments') as b:
  b.print(pargs, indent=False)

for (epoch, batch, steps), data in trainer(train_dataset, epochs=pargs.epochs, progress='Training', batch_size=pargs.batch_size):
  optimizer.zero_grad()
  model.train()
  
  loss_value, out = model(*data)

  loss_value.backward()
  optimizer.step()

  if steps % pargs.log_frequency == 0:
    trainer.log(steps,
                loss=loss_value,
                model=model.log(loss=loss_value, 
                                out=out,
                                data=data,
                                epoch=epoch,
                                batch=batch,
                                step=steps))

  if (steps + 1) % 100000 == 0:
    model.eval()
    validation_loss = sum(model(*data)[0].item() for _, data in trainer(validation_dataset, progress='Validation', leave=False)) 
    validation_loss /= len(validation_dataset)
    trainer.log(steps, validation=validation_loss)

  trainer.checkpoint()
    
print()