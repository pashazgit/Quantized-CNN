
def reset_lr(scheduler):
    """Reset the learning rate scheduler.
    """
    scheduler.base_lrs = list(map(lambda group: group['initial_lr'], scheduler.optimizer.param_groups))
    last_epoch = 0
    scheduler.last_epoch = last_epoch
    scheduler.step(last_epoch)

