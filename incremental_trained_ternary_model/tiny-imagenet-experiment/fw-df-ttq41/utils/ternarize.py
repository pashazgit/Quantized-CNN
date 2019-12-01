
import torch


def ternarize(opt, opt_fp, opt_sf, thr):
    with torch.no_grad():
        for group, group_fp, group_sf in zip(opt.param_groups[0:4], opt_fp.param_groups, opt_sf.param_groups):
            params_fp = group_fp['params'][0]
            delta = torch.max(params_fp.abs(), dim=-1, keepdim=True)[0] * thr
            a = (params_fp > delta).float()
            b = (params_fp < -delta).float()
            a_s, b_s = group_sf['params']
            c = (a * a_s + b * b_s).reshape(len(group['params']), *group['params'][0].size())
            for _i, param in enumerate(group['params']):
                # a leaf Variable that requires grad cannot be used in an in-place operation.
                param.data[:] = c[_i]

