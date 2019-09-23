
import numpy as np
import torch


def inq_step(opt_fp, steps, inq_idx, strategy='pruning'):
    """Handles the the weight partitioning of the incremental network ternarization procedure.
    Args:
        opt (Optimizer): Wrapped optimizer.
        steps (list): accumulated portions of ternarized weights.
        inq_idx (int): current round index
        strategy ("random"|"pruning"): weight partition strategy, either random or pruning-inspired.
    """
    for group_fp in opt_fp.param_groups:
        params_fp = group_fp['params'][0].cpu()
        zeros = torch.zeros_like(params_fp)
        ones = torch.ones_like(params_fp)
        q = torch.tensor(
            np.quantile(torch.abs(params_fp).numpy(), 1 - steps[inq_idx], axis=-1, keepdims=True),
            dtype=torch.float32)
        if torch.cuda.is_available():
            group_fp['part'] = torch.where(torch.abs(params_fp) >= q, ones, zeros).cuda()
        else:
            group_fp['part'] = torch.where(torch.abs(params_fp) >= q, ones, zeros)



