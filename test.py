import wandb

wandb.init(project='intentDICE', name='test', mode='online')

i = 0
while i < 200:
    if i % 10 == 0:
        wandb.log({'a': i}, step = i)
    if i % 25 == 0:
        wandb.log({'b': i}, step = i)
    i += 1
