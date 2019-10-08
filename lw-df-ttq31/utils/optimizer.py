
import torch
from torch import optim


def create_optimizer(model, lr=None):
    conv0_params = [param for name, param in model.named_parameters() if 'conv' in name and 'weight' in name
                    and param.size() == (64, 64, 3, 3)]
    conv1_params = [param for name, param in model.named_parameters() if 'conv' in name and 'weight' in name
                    and param.size() == (128, 128, 3, 3)]
    conv2_params = [param for name, param in model.named_parameters() if 'conv' in name and 'weight' in name
                    and param.size() == (256, 256, 3, 3)]
    conv3_params = [param for name, param in model.named_parameters() if 'conv' in name and 'weight' in name
                    and param.size() == (512, 512, 3, 3)]

    x_params = [param for name, param in model.named_parameters() if 'conv' not in name]

    groups = [{'params': conv0_params, 'name': 'conv0_params'}, {'params': conv1_params, 'name': 'conv1_params'},
              {'params': conv2_params, 'name': 'conv2_params'}, {'params': conv3_params, 'name': 'conv3_params'},
              {'params': x_params, 'name': 'x_params'}]

    optimizer = optim.Adam(groups, lr=lr)

    return optimizer


def copy_optimizer(opt, lr=None):
    groups_fp = []
    for group in opt.param_groups[0:4]:
        # Notice: param.data.clone() is used instead of param.clone() because it can't optimize non-leaf Tensor
        a = torch.cat([param.data.clone().reshape(1, -1) for param in group['params']])
        groups_fp.append({'params': a})

    optimizer_fp = optim.Adam(groups_fp, lr=lr)

    return optimizer_fp
