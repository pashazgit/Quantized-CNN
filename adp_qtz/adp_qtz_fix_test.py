4
# Copyright (C), Visual Computing Group @ University of Victoria.
# Copyright (C), https://github.com/pytorch/examples/blob/master/imagenet/main.py
# Copyright (C), https://github.com/TropComplique/trained-ternary-quantization
# Copyright (C), https://github.com/Mxbonn/INQ-pytorch
# Copyright (C), https://github.com/vinsis/ternary-quantization


import pdb
import sys
# sys.path.insert(0, "./adaptive_quantization/input_pipeline")
# sys.path.insert(0, "./adaptive_quantization/utils")
import os
import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torchvision
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import h5py
import torch.nn.functional as F
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.nn import init
import math
import pdb


# ----------------------------------------
# Global variables within this script
arg_lists = []
parser = argparse.ArgumentParser()


# ----------------------------------------
# Some nice macros to be used for arparse
def str2bool(v):
    return v.lower() in ("true", "1")


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# -----------------------------------------
parser.add_argument("--mode", type=str,
                    default="train",
                    choices=["train", "test"],
                    help="Run mode")

parser.add_argument("--data_dir", type=str,
                    default="/home/pasha/scratch/datasets/cifar-10-batches-py",
                    help="")

parser.add_argument("--save_dir", type=str,
                    default="/home/pasha/scratch/jobs/output/adp_qtz/fix/saves",
                    help="Directory to save the best model")

parser.add_argument("--save_dir_b", type=str,
                    default="/home/pasha/scratch/jobs/output/adp_qtz/baseline/saves",
                    help="Directory to save the best model")

parser.add_argument("--log_dir", type=str,
                    default="/home/pasha/scratch/jobs/output/adp_qtz/fix/logs",
                    help="Directory to save logs and current model")

# parser.add_argument("--plt_dir", type=str,
#                     default="/home/pasha/scratch/adaptive_quantization/jobs/output/adp_qtz/fix/plts",
#                     help="Directory to save logs and current model")

parser.add_argument("--batch_size", type=int,
                    default=128,
                    help="Size of each training batch")

# parser.add_argument("--l2_reg", type=float,
#                     default=5e-4,
#                     help="L2 Regularization strength")

# parser.add_argument("--en_loss", type=float,
#                     default=0,
#                     help="sharpening hyper_parameter")

parser.add_argument("--num_level_conv", type=int,
                    default=32,
                    help="number of quantization levels for conv layer")

parser.add_argument("--num_level_fc", type=int,
                    default=32,
                    help="number of quantization levels for fc layer")

parser.add_argument("--prim_init", type=str,
                    default="uniform",
                    choices=["uniform", "normal"],
                    help="init mode of prim coefficients")

parser.add_argument("--liar", type=str2bool,
                    default=True,
                    help="Whether to round or lower the range of the weights within a layer")

parser.add_argument("--name_idx", type=int,
                    default=0,
                    help="")

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


def print_usage():
    parser.print_usage()


def test(config):

    global beta  # sharpening factor

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4821, 0.4462), (0.2472, 0.2437, 0.2617)),
    ])

    # Initialize Dataset for testing.
    test_data = CIFAR10Dataset(config.data_dir, mode="test", transform=transform_test)
    # Create data loader for the test dataset with two number of workers and no shuffling.

    test_loader = DataLoader(
        dataset=test_data,
        batch_size=100,
        num_workers=2,
        shuffle=False)

    # Create model
    model = ResNet(config, 3)
    print('\nmodel created')
    # Move model to gpu if cuda is available
    if torch.cuda.is_available():
        # model = torch.nn.DataParallel(model).cuda()
        model = model.cuda()

    # Define the loss function (criterion)
    criterion = nn.CrossEntropyLoss()
    # Move criterion to gpu if cuda is available
    if torch.cuda.is_available():
        criterion = criterion.cuda()

    # Load our best model and set model for testing
    bestmodel_file = os.path.join(config.save_dir, "bestmodel_{}.pth".format(config.name_idx))
    checkpoint_file = os.path.join(config.save_dir, "checkpoint_{}.pth".format(config.name_idx))
    if config.mode == 'best':
        load_res = torch.load(
            bestmodel_file,
            map_location="cpu")
        beta = load_res["beta"]
        model.load_state_dict(load_res["model"])
    elif config.mode == 'check':
        load_res = torch.load(
            checkpoint_file,
            map_location="cpu")
        beta = load_res["beta"]
        model.load_state_dict(load_res["model"])
    else:
        raise ValueError("Unknown load mode \"{}\"".format(config.mode))

    model.eval()
    # Implement The Test loop
    prefix = "Testing: "
    # List to contain all losses and accuracies for all the val batches
    test_loss = []
    test_acc = []
    for x, y in tqdm(test_loader, desc=prefix):
        # Send data to GPU if we have one
        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()
        # Don't invoke gradient computation
        with torch.no_grad():
            # Compute logits
            logits = model.forward(x)
            # Compute accuracy and store as numpy
            pred = torch.argmax(logits, dim=1)
            acc = torch.mean(torch.eq(pred, y).float()) * 100.0
            test_acc += [acc.cpu().numpy()]
    # Take average
    test_acc_avg = np.mean(test_acc)

    # Report Test loss and accuracy
    print("test_acc: ", test_acc_avg)


def unpickle(file_name):
    with open(file_name, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_data(data_dir, data_type):
    if data_type == "train":
        data = []
        label = []
        for _i in range(4):
            file_name = os.path.join(data_dir, "data_batch_{}".format(_i + 1))
            cur_dict = unpickle(file_name)
            data += [np.array(cur_dict[b"data"])]
            label += [np.array(cur_dict[b"labels"])]
        cur_dict = unpickle(os.path.join(data_dir, "data_batch_5"))
        cur_dict_data = np.array(cur_dict[b"data"])
        cur_dict_labels = np.array(cur_dict[b"labels"])
        cur_dict_data_f = cur_dict_data[0:int(len(cur_dict_data) / 2)]
        cur_dict_labels_f = cur_dict_labels[0:int(len(cur_dict_labels) / 2)]
        data += [cur_dict_data_f]
        label += [cur_dict_labels_f]
        # Concat them
        data = np.concatenate(data)
        label = np.concatenate(label)

    elif data_type == "valid":
        # We'll use the 5th batch as our validation data. Note that this is not
        # the best way to do this. One strategy I've seen is to use this to
        # figure out the loss value you should aim to train for, and then stop
        # at that point, using the entire dataset. However, for simplicity,
        # we'll use this simple strategy.
        cur_dict = unpickle(os.path.join(data_dir, "data_batch_5"))
        cur_dict_data = np.array(cur_dict[b"data"])
        cur_dict_labels = np.array(cur_dict[b"labels"])
        data = cur_dict_data[int(len(cur_dict_data) / 2):]
        label = cur_dict_labels[int(len(cur_dict_labels) / 2):]

    elif data_type == "test":
        cur_dict = unpickle(os.path.join(data_dir, "test_batch"))
        data = np.array(cur_dict[b"data"])
        label = np.array(cur_dict[b"labels"])

    else:
        raise ValueError("Wrong data type {}".format(data_type))

    # Turn data into (NxCxHxW) format, so that we can easily process it, where
    # N=number of images, H=height, W=widht, C=channels. Note that this
    # is the default format for PyTorch.
    data = np.transpose(np.reshape(data, (-1, 3, 32, 32)), (0, 2, 3, 1))

    return data, label


class CIFAR10Dataset(Dataset):
    def __init__(self, data_dir, mode, transform):
        print("Loading CIFAR10 Dataset from {} for {}ing ...".format(data_dir, mode), end="")
        # Load data (note that we now simply load the raw data.
        data, label = load_data(data_dir, mode)
        self.data = data
        self.label = label
        self.transform = transform
        print(" done.")

    def __getitem__(self, index):
        # Grab one data from the dataset and then apply our feature extraction
        data_cur = self.data[index]
        # doing this so that it is consistent with all other datasets to return a PIL Image
        data_cur = Image.fromarray(data_cur)
        # Make pytorch object
        if self.transform is not None:
            data_cur = self.transform(data_cur)
        # Label is just the label
        label_cur = self.label[index]
        return data_cur, label_cur

    def __len__(self):
        # return the number of elements at `self.data`
        return len(self.data)


class Residual_b(nn.Module):
    def __init__(self, in_channel, increase_dim):
        super(Residual_b, self).__init__()
        if not increase_dim:
            out_channel = in_channel
            stride = 1
        else:
            out_channel = in_channel * 2
            stride = 2

        self.bn1 = nn.BatchNorm2d(in_channel)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)

        if not increase_dim:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(2),
                nn.ZeroPad2d((0, 0, 0, 0, in_channel//2, in_channel//2)))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = F.relu(self.bn2(self.conv1(out)))
        out = self.conv2(out)
        out += self.shortcut(x)
        return out


class PreResidual_b(nn.Module):
    def __init__(self, in_channel, increase_dim=False):
        super(PreResidual_b, self).__init__()
        out_channel = in_channel
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = F.relu(self.bn2(self.conv1(x)))
        out = (self.conv2(out))
        out += x
        return out


class ResNet_b(nn.Module):
    def __init__(self, n):
        super(ResNet_b, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(n, in_plane=16, first=True)
        # 32, c = 16
        self.layer2 = self._make_layer(n, in_plane=16, first=False)
        # 16, c = 32
        self.layer3 = self._make_layer(n, in_plane=32, first=False)
        # 8, c = 64
        self.bnlast = nn.BatchNorm2d(64)
        self.ap = nn.AvgPool2d(8)
        self.linear = nn.Linear(64, 10)

    def _make_layer(self, n, in_plane, first):
        layers = []

        if not first:
            layers.append(Residual_b(in_channel=in_plane, increase_dim=True))
            in_plane *= 2
        else:
            layers.append(PreResidual_b(in_channel=in_plane, increase_dim=False))

        for k in range(1, n):
            layers.append(Residual_b(in_channel=in_plane, increase_dim=False))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn0(self.conv0(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bnlast(out))
        out = self.ap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class MyConv2d(nn.Module):
    def __init__(self, config, s, in_channel, out_channel, ksize, stride, padding, bias=False):
        super(MyConv2d, self).__init__()
        # primary coefficients initialization
        if config.prim_init == 'uniform':
            self.p_c = nn.Parameter(torch.rand((out_channel, in_channel, ksize, ksize, config.num_level_conv-1)), requires_grad=True)
        elif config.prim_init == 'normal':
            self.p_c = nn.Parameter(torch.randn((out_channel, in_channel, ksize, ksize, config.num_level_conv-1)), requires_grad=True)
        self.q_level = nn.Parameter(torch.Tensor(config.num_level_conv-1), requires_grad=False)  # quantization levels
        # self.bias = nn.Parameter(torch.randn((out_channel,)), requires_grad=True)
        self.ksize = ksize
        self.stride = stride
        self.padding = padding
        self.mode = config.mode
        # Our custom convolution kernel. We'll initialize it by fixed quantization levels
        self.reset_parameters(config.num_level_conv, s)

    def reset_parameters(self, num_level, s):
        t1 = num_level / 2
        t2 = math.floor(math.log(s, 2)) if config.liar else round(math.log(s, 2))
        t3 = torch.arange(t2 - t1 + 2, t2 + 1)
        self.q_level.data[:] = torch.cat(((-2 ** t3).sort()[0], torch.zeros(1), 2 ** t3))

    def forward(self, x):
        p_c_norm = torch.sqrt(torch.sum(self.p_c**2, dim=-1, keepdim=True))
        p_c_normal = self.p_c / p_c_norm
        s_c = torch.exp(beta * p_c_normal) / \
            torch.sum(torch.exp(beta * p_c_normal), dim=-1, keepdim=True)  # the secondary coefficients
        _, idx = torch.max(s_c, dim=-1)
        conv_w = torch.index_select(self.q_level, -1, idx.reshape(-1)).reshape(*idx.size())  # the conv layer's weights
        assert len(x.shape) == 4
        pad = nn.ZeroPad2d(padding=self.padding)
        x = pad(x)
        x_out = None
        h, w = x.shape[-2:]
        h = h - self.ksize + 1
        w = w - self.ksize + 1
        for _i in range(self.ksize):
            for _j in range(self.ksize):
                # Get the region that we will apply for this part of the convolution kernel
                cur_x = x[:, :, range(_j, _j + h, self.stride), :][:, :, :, range(_i, _i + w, self.stride)]
                h_n = cur_x.shape[2]
                w_n = cur_x.shape[3]
                # Make matrix multiplication ready
                cur_x = cur_x.reshape(x.shape[0], x.shape[1], h_n * w_n)
                # Get the current multiplication kernel
                cur_w = conv_w[:, :, _j, _i]
                # Left multiply
                cur_o = torch.matmul(cur_w, cur_x)
                # Return to original shape
                cur_o = cur_o.reshape(x.shape[0], conv_w.shape[0], h_n, w_n)
                # Cumulative sum
                if x_out is None:
                    x_out = cur_o
                else:
                    x_out += cur_o
        # x_out += self.bias.view(1, self.bias.shape[0], 1, 1)
        return x_out


class MyLinear(nn.Module):
    def __init__(self, config, s, v, in_feature, out_feature, bias=True):
        super(MyLinear, self).__init__()
        # primary coefficients initialization
        if config.prim_init == 'uniform':
            self.p_c = nn.Parameter(torch.rand((in_feature, out_feature, config.num_level_fc-1)), requires_grad=True)
        elif config.prim_init == 'normal':
            self.p_c = nn.Parameter(torch.randn((in_feature, out_feature, config.num_level_fc-1)), requires_grad=True)
        self.q_level = nn.Parameter(torch.Tensor(config.num_level_fc-1), requires_grad=False)  # quantization levels
        self.bias = nn.Parameter(torch.Tensor(out_feature), requires_grad=True)
        self.mode = config.mode
        # Our custom linear layer. We'll initialize it by fixed quantization levels
        self.reset_parameters(config.num_level_fc, s, v)

    def reset_parameters(self, num_level, s, v):
        t1 = num_level / 2
        t2 = math.floor(math.log(s, 2)) if config.liar else round(math.log(s, 2))
        t3 = torch.arange(t2 - t1 + 2, t2 + 1)
        self.q_level.data[:] = torch.cat(((-2 ** t3).sort()[0], torch.zeros(1), 2 ** t3))

        with torch.no_grad():
            self.bias.data[:] = v.data

    def forward(self, x):
        p_c_norm = torch.sqrt(torch.sum(self.p_c ** 2, dim=-1, keepdim=True))
        p_c_normal = self.p_c / p_c_norm
        s_c = torch.exp(beta * p_c_normal) / \
              torch.sum(torch.exp(beta * p_c_normal), dim=-1, keepdim=True)  # the secondary coefficients
        _, idx = torch.max(s_c, dim=-1)
        linear_w = torch.index_select(self.q_level, -1, idx.reshape(-1)).reshape(*idx.size())  # the linear layer's weights
        assert len(x.shape) == 2
        x_out = torch.matmul(x, linear_w) + self.bias
        return x_out


class Residual(nn.Module):
    def __init__(self, config, s1, s2, in_channel, increase_dim):
        super(Residual, self).__init__()
        if not increase_dim:
            out_channel = in_channel
            stride = 1
        else:
            out_channel = in_channel * 2
            stride = 2

        self.bn1 = nn.BatchNorm2d(in_channel)
        self.conv1 = MyConv2d(config, s1, in_channel, out_channel, ksize=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv2 = MyConv2d(config, s2, out_channel, out_channel, ksize=3, stride=1, padding=1, bias=False)

        if not increase_dim:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(2),
                nn.ZeroPad2d((0, 0, 0, 0, in_channel//2, in_channel//2)))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = F.relu(self.bn2(self.conv1(out)))
        out = self.conv2(out)
        out += self.shortcut(x)
        return out


class PreResidual(nn.Module):
    def __init__(self, config, s1, s2, in_channel, increase_dim=False):
        super(PreResidual, self).__init__()
        out_channel = in_channel
        self.conv1 = MyConv2d(config, s1, in_channel, out_channel, ksize=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv2 = MyConv2d(config, s2, out_channel, out_channel, ksize=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = F.relu(self.bn2(self.conv1(x)))
        out = (self.conv2(out))
        out += x
        return out


class ResNet(nn.Module):
    def __init__(self, config, s, v):
        super(ResNet, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(16)
        layers = []
        layers.append(PreResidual(config, s[1], s[2], in_channel=16, increase_dim=False))
        layers.append(Residual(config, s[3], s[4], in_channel=16, increase_dim=False))
        layers.append(Residual(config, s[5], s[6], in_channel=16, increase_dim=False))
        self.layer1 = nn.Sequential(*layers)
        # 32, c = 16
        layers = []
        layers.append(Residual(config, s[7], s[8], in_channel=16, increase_dim=True))
        layers.append(Residual(config, s[9], s[10], in_channel=32, increase_dim=False))
        layers.append(Residual(config, s[11], s[12], in_channel=32, increase_dim=False))
        self.layer2 = nn.Sequential(*layers)
        # 16, c = 32
        layers = []
        layers.append(Residual(config, s[13], s[14], in_channel=32, increase_dim=True))
        layers.append(Residual(config, s[15], s[16], in_channel=64, increase_dim=False))
        layers.append(Residual(config, s[17], s[18], in_channel=64, increase_dim=False))
        self.layer3 = nn.Sequential(*layers)
        # 8, c = 64
        self.bnlast = nn.BatchNorm2d(64)
        self.ap = nn.AvgPool2d(8)
        self.linear = MyLinear(config, s[19], v, in_feature=64, out_feature=10, bias=True)

    def forward(self, x):
        out = F.relu(self.bn0(self.conv0(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bnlast(out))
        out = self.ap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# def model_loss(config, beta, model):
#     loss = 0
#     p_c = None
#     for name, param in model.named_parameters():
#         if "weight" in name:
#             loss += torch.sum(param**2)
#         if 'p_c' in name:
#             p_c = param
#         if 'q_level' in name:
#             q_level = param
#             p_c_norm = torch.sqrt(torch.sum(p_c ** 2, dim=-1, keepdim=True))
#             p_c_normal = p_c / p_c_norm
#             s_c = torch.exp(beta * p_c_normal) / \
#                 torch.sum(torch.exp(beta * p_c_normal), dim=-1, keepdim=True)  # the secondary coefficients
#             weight = torch.matmul(s_c, q_level)  # the conv and linear layer's weights
#             loss += torch.sum(weight ** 2)
#
#     return config.l2_reg * loss


# def entropy_loss(config, beta, model):
#     loss = 0
#     for name, param in model.named_parameters():
#         if 'p_c' in name:
#             p_c = param
#             p_c_norm = torch.sqrt(torch.sum(p_c ** 2, dim=-1, keepdim=True))
#             p_c_normal = p_c / p_c_norm
#             s_c = torch.exp(beta * p_c_normal) / \
#                 torch.sum(torch.exp(beta * p_c_normal), dim=-1, keepdim=True)  # the secondary coefficients
#             loss -= torch.sum(s_c * torch.log(s_c))
#
#     return config.sharp * loss


# def plot_entropy(config, beta, iter_idx, model):
#     # plt.style.use('seaborn-white')
#     mpl.use('Agg')
#     fig = plt.figure()
#
#     e_conv = []  # entropy of conv layers
#     for name, param in model.named_parameters():
#         if 'p_c' in name:
#             if 'conv' in name:
#                 p_c = param
#                 p_c_norm = torch.sqrt(torch.sum(p_c ** 2, dim=-1, keepdim=True))
#                 p_c_normal = p_c / p_c_norm
#                 s_c = torch.exp(beta * p_c_normal) / \
#                       torch.sum(torch.exp(beta * p_c_normal), dim=-1, keepdim=True)  # the secondary coefficients
#                 e_conv.append(torch.sum(-s_c * torch.log(s_c), dim=-1).reshape(-1).cpu())
#             elif 'linear' in name:
#                 p_c = param
#                 p_c_norm = torch.sqrt(torch.sum(p_c ** 2, dim=-1, keepdim=True))
#                 p_c_normal = p_c / p_c_norm
#                 s_c = torch.exp(beta * p_c_normal) / \
#                       torch.sum(torch.exp(beta * p_c_normal), dim=-1, keepdim=True)  # the secondary coefficients
#                 e_linear = torch.sum(-s_c * torch.log(s_c), dim=-1).reshape(-1).cpu()  # entropy of linear layer
#
#     e_conv = torch.cat(e_conv)
#     plt.hist(e_conv.detach().numpy())
#     fig.savefig(os.path.join(config.plt_dir, 'entropy_conv_{}'.format(config.name_idx), 'fig_{}.png'.format(iter_idx)))
#
#     plt.hist(e_linear.detach().numpy())
#     fig.savefig(os.path.join(config.plt_dir, 'entropy_fc_{}'.format(config.name_idx), 'fig_{}.png'.format(iter_idx)))


if __name__ == "__main__":

    # ----------------------------------------
    # Parse configuration
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    test(config)
