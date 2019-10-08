
from pdb import set_trace as bp
import torch


def get_grads(opt, opt_fp, opt_sf, thr):
    with torch.no_grad():
        for group, group_fp, group_sf in zip(opt.param_groups[0:4], opt_fp.param_groups, opt_sf.param_groups):
            params_grad = torch.cat([param.grad for param in group['params']])
            params_grad_v = params_grad.view(params_grad.size(0), -1)
            params_fp = group_fp['params'][0]
            delta = torch.max(params_fp.abs(), dim=-1, keepdim=True)[0] * thr
            a = (params_fp > delta).float()
            b = (params_fp < -delta).float()
            c = 1 - a - b
            # params_fp.grad = group_sf['params'][0] * a * params_grad_v - group_sf['params'][1] * b * params_grad_v + \
            #           c * params_grad_v
            params_fp.grad = a * params_grad_v + b * params_grad_v + \
                             c * params_grad_v
            # group_sf['params'][0].grad = (a * params_grad_v).sum(dim=-1, keepdim=True)/a.sum(dim=-1, keepdim=True)
            # group_sf['params'][1].grad = (b * params_grad_v).sum(dim=-1, keepdim=True)/b.sum(dim=-1, keepdim=True)
            group_sf['params'][0].grad = (a * params_grad_v).sum(dim=-1, keepdim=True)
            group_sf['params'][1].grad = (b * params_grad_v).sum(dim=-1, keepdim=True)
