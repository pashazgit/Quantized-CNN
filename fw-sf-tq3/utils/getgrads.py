
from pdb import set_trace as bp
import torch


def get_grads(opt, opt_fp):
    with torch.no_grad():
        for group, group_fp in zip(opt.param_groups[0:4], opt_fp.param_groups):
            params_grad = torch.cat([param.grad for param in group['params']])
            params_grad_v = params_grad.view(params_grad.size(0), -1)
            params_fp = group_fp['params'][0]
            params_fp.grad = params_grad_v.data
