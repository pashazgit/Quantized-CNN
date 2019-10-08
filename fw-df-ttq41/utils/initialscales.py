
import torch


def initial_scales(opt_fp, thr):
    """Compute initial scaling factors at the beginning of each round.
        Args:
            opt (Optimizer): Wrapped optimizer.
        """
    sf = []
    for group_fp in opt_fp.param_groups:
        sf_g = []
        # Notice: tensor.data.clone() is used instead of tensor.clone() because it can't optimize non-leaf Tensor
        params_fp = group_fp['params'][0].data.clone()
        delta = torch.max(params_fp.abs(), dim=-1, keepdim=True)[0] * thr
        a_p = (params_fp > delta).float() * params_fp
        # b_p = (params_fp > delta).float()
        # w_p = a_p.sum(dim=-1, keepdim=True)/b_p.sum(dim=-1, keepdim=True)
        w_p = a_p.mean(dim=-1, keepdim=True)
        w_p.requires_grad = True
        if torch.cuda.is_available():
            w_p = w_p.cuda()
        sf_g.append(w_p)
        a_n = (params_fp < -delta).float() * params_fp
        # b_n = (params_fp < -delta).float()
        # w_n = a_n.sum(dim=-1, keepdim=True)/b_n.sum(dim=-1, keepdim=True)
        w_n = a_n.mean(dim=-1, keepdim=True)
        w_n.requires_grad = True
        if torch.cuda.is_available():
            w_n = w_n.cuda()
        sf_g.append(w_n)

        sf.append({'params': sf_g})

    return sf

