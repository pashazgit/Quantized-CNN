
import torch


def ternarize(opt, opt_fp):
    with torch.no_grad():
        for group, group_fp in zip(opt.param_groups[0:4], opt_fp.param_groups):
            params_fp = group_fp['params'][0]
            delta = 0.7 * torch.mean(params_fp.abs(), dim=-1, keepdim=True)
            a = (params_fp.abs() > delta).float()
            alpha = (a * params_fp.abs()).sum(dim=-1, keepdim=True)/a.sum(dim=-1, keepdim=True)
            b = (params_fp > delta).float()
            c = (params_fp < -delta).float()
            d = (b * alpha - c * alpha).reshape(len(group['params']), *group['params'][0].size())
            for _i, param in enumerate(group['params']):
                # a leaf Variable that requires grad cannot be used in an in-place operation.
                param.data[:] = d[_i]












