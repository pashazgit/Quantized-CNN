4
# Copyright (C), Visual Computing Group @ University of Victoria.
# Copyright (C), https://github.com/pytorch/examples/blob/master/imagenet/main.py
# Copyright (C), https://github.com/TropComplique/trained-ternary-quantization
# Copyright (C), https://github.com/Mxbonn/INQ-pytorch
# Copyright (C), https://github.com/vinsis/ternary-quantization


from pdb import set_trace as bp
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

# parser.add_argument("--data_dir", type=str,
#                     default="/home/mostafa/Downloads/datasets/cifar-10-batches-py",
#                     help="")

parser.add_argument("--save_dir", type=str,
                    default="/home/pasha/scratch/adaptive_quantization/jobs/output/saves",
                    help="Directory to save the best model")

# parser.add_argument("--save_dir", type=str,
#                     default="/home/mostafa/Uvic/Thesis/DL-compression/implementation/adaptive_quantization/jobs/output/saves",
#                     help="Directory to save the best model")

parser.add_argument("--log_dir", type=str,
                    default="/home/pasha/scratch/adaptive_quantization/jobs/output/logs",
                    help="Directory to save logs and current model")

# parser.add_argument("--log_dir", type=str,
#                     default="/home/mostafa/Uvic/Thesis/DL-compression/implementation/adaptive_quantization/jobs/output/logs",
#                     help="Directory to save logs and current model")

parser.add_argument("--learning_rate", type=float,
                    default=1e-3,
                    help="Learning rate (gradient step size)")

parser.add_argument("--batch_size", type=int,
                    default=100,
                    help="Size of each training batch")

parser.add_argument("--num_epoch", type=int,
                    default=60,
                    help="Number of epochs to train")

parser.add_argument("--val_intv", type=int,
                    default=400,
                    help="Validation interval")

parser.add_argument("--rep_intv", type=int,
                    default=400,
                    help="Report interval")

parser.add_argument("--weight_decay", type=float,
                    default=1e-4,
                    help="L2 Regularization strength")

parser.add_argument("--resume", type=str2bool,
                    default=False,
                    help="Whether to resume training from existing checkpoint")


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


def print_usage():
    parser.print_usage()


def main(config):
    if config.mode == "train":
        train(config)
    elif config.mode == "test":
        test(config)
    else:
        raise ValueError("Unknown run mode \"{}\"".format(config.mode))


def train(config):
    # CUDA_LAUNCH_BLOCKING = 1

    # Initialize datasets for both training and validation
    train_data = CIFAR10Dataset(
        config.data_dir, mode="train")
    val_data = CIFAR10Dataset(
        config.data_dir, mode="valid")
    # Create data loader for training and validation.
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=config.batch_size,
        num_workers=2,
        shuffle=True)
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=config.batch_size,
        num_workers=2,
        shuffle=False)

    # Create model instance.
    model = ResNet(config, PreActBlock, [2, 2, 2, 2], num_classes=10)
    print('\nmodel created')
    # Move model to gpu if cuda is available
    if torch.cuda.is_available():
        # model = torch.nn.DataParallel(model).cuda()
        model = model.cuda()
    model.train()

    # Define the loss function (criterion)
    criterion = nn.CrossEntropyLoss()
    # Move criterion to gpu if cuda is available
    if torch.cuda.is_available():
        criterion = criterion.cuda()

    # create optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Create log directory and save directory if it does not exist
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    # Create summary writer
    train_writer = SummaryWriter(
        log_dir=os.path.join(config.log_dir, "train_adp_qtz_baseline"))
    val_writer = SummaryWriter(
        log_dir=os.path.join(config.log_dir, "valid_adp_qtz_baseline"))

    # Initialize training
    start_epoch = 0
    iter_idx = -1  # make counter start at zero
    best_val_acc1 = 0  # to check if best validation accuracy
    # Prepare checkpoint file and model file to save and load from
    checkpoint_file = os.path.join(config.save_dir, "checkpoint_adp_qtz_baseline.pth")
    bestmodel_file = os.path.join(config.save_dir, "bestmodel_adp_qtz_baseline.pth")

    # Check for existing training results. If it existst, and the configuration
    # is set to resume `config.resume==True`, resume from previous training. If
    # not, delete existing checkpoint.
    if os.path.exists(checkpoint_file):
        if config.resume:
            # Use `torch.load` to load the checkpoint file and the load the
            # things that are required to continue training. For the model and
            # the optimizer, use `load_state_dict`. It's actually a good idea
            # to code the saving part first and then code this part.
            print("Checkpoint found! Resuming")
            # Read checkpoint file.
            load_res = torch.load(
                checkpoint_file,
                map_location="cpu")
            start_epoch = load_res["epoch"]
            # Resume iterations
            iter_idx = load_res["iter_idx"]
            # Resume best va result
            best_val_acc1 = load_res["best_val_acc1"]
            # Resume model
            model.load_state_dict(load_res["model"])
            # Resume optimizer
            optimizer.load_state_dict(load_res["optimizer"])
        else:
            os.remove(checkpoint_file)

    # Training loop
    for epoch in tqdm(range(start_epoch, config.num_epoch)):
        # print("adp_qtz_baseline_epoch: ", epoch)
        if epoch == 20:
            optimizer.param_groups[0]['lr'] /= 10

        for x, y in train_loader:
            # print("adp_qtz_baseline_iter_idx: ", iter_idx)
            iter_idx += 1  # Counter
            # Send data to GPU if we have one
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            # Apply the model to obtain scores (forward pass)
            logits = model.forward(x)
            # Compute the loss
            loss = criterion(logits, y) + config.weight_decay * model_loss(config, model)
            # Compute gradients
            loss.backward()
            optimizer.step()
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Monitor results every report interval
            if iter_idx % config.rep_intv == 0:
                # print('adp_qtz_baseline_iter_idx: ', iter_idx)
                # compute accuracies
                acc1, acc5 = accuracy(logits, y, topk=(1, 5))
                # Write loss and accuracy to tensorboard, using keywords `loss` and `accuracy`.
                train_writer.add_scalar("loss", loss, global_step=iter_idx)
                train_writer.add_scalar("top1 accuracy", acc1, global_step=iter_idx)
                train_writer.add_scalar("top5 accuracy", acc5, global_step=iter_idx)
                # Save
                torch.save({
                    "epoch": epoch,
                    "iter_idx": iter_idx,
                    "best_val_acc1": best_val_acc1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict()
                }, checkpoint_file)

            # Validate results every validation interval
            if iter_idx % config.val_intv == 0:
                # Set model for evaluation
                model.eval()
                # List to contain all losses and accuracies for all the val batches
                val_loss = []
                val_acc1 = []
                val_acc5 = []
                for x, y in val_loader:
                    # Send data to GPU if we have one
                    if torch.cuda.is_available():
                        x, y = x.cuda(), y.cuda()
                    # Apply forward pass to compute the losses and accuracies for each of the val batches
                    with torch.no_grad():
                        # Compute logits
                        logits = model.forward(x)
                        # Compute loss and store as numpy
                        loss = criterion(logits, y) + config.weight_decay * model_loss(config, model)
                        # print(torch.cuda.is_available())
                        val_loss += [loss.cpu()]
                        # Compute accuracy and store as numpy
                        acc1, acc5 = accuracy(logits, y, topk=(1, 5))
                        val_acc1 += [acc1.data.cpu()]
                        val_acc5 += [acc5.data.cpu()]
                # Take average
                val_loss_avg = np.mean(val_loss)
                val_acc1_avg = np.mean(val_acc1)
                val_acc5_avg = np.mean(val_acc5)
                # Write loss and accuracy to tensorboard, using keywords `loss` and `accuracy`.
                val_writer.add_scalar("loss", val_loss_avg, global_step=iter_idx)
                val_writer.add_scalar("top1 accuracy", val_acc1_avg, global_step=iter_idx)
                val_writer.add_scalar("top5 accuracy", val_acc5_avg, global_step=iter_idx)
                # Set model back for training
                model.train()
                if val_acc1_avg > best_val_acc1:
                    best_val_acc1 = val_acc1_avg
                    # Save
                    torch.save({
                        "epoch": epoch,
                        "iter_idx": iter_idx,
                        "best_val_acc1": best_val_acc1,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict()
                    }, bestmodel_file)

    print("adp_qtz_baseline_best_val_acc1: ", best_val_acc1)


def test(config):
    # Initialize Dataset for testing.
    test_data = CIFAR10Dataset(
        config, mode="test")
    # Create data loader for the test dataset with two number of workers and no shuffling.
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=config.batch_size,
        num_workers=2,
        shuffle=False)

    # Create model
    model = ResNet(config, PreActBlock, [2, 2, 2, 2], num_classes=10)
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
    bestmodel_file = os.path.join(config.save_dir, "bestmodel_adp_qtz_baseline.pth")
    load_res = torch.load(
        bestmodel_file,
        map_location="cpu")
    model.load_state_dict(load_res["model"])

    model.eval()
    # Implement The Test loop
    prefix = "Testing: "
    # List to contain all losses and accuracies for all the val batches
    test_loss = []
    test_acc1 = []
    test_acc5 = []
    for x, y in tqdm(test_loader, desc=prefix):
        # Send data to GPU if we have one
        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()
        # Don't invoke gradient computation
        with torch.no_grad():
            # Compute logits
            logits = model.forward(x)
            # Compute loss and store as numpy
            loss = criterion(logits, y) + config.weight_decay * model_loss(config, model)
            # print(torch.cuda.is_available())
            test_loss += [loss.cpu()]
            # Compute accuracy and store as numpy
            acc1, acc5 = accuracy(logits, y, topk=(1, 5))
            test_acc1 += [acc1.data.cpu()]
            test_acc5 += [acc5.data.cpu()]
    # Take average
    test_loss_avg = np.mean(test_loss)
    test_acc1_avg = np.mean(test_acc1)
    test_acc5_avg = np.mean(test_acc5)

    # Report Test loss and accuracy
    print("adp_qtz_baseline_test_loss: ", test_loss_avg)
    print("adp_qtz_baseline_test_acc1: ", test_acc1_avg)


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
            data += [
                np.array(cur_dict[b"data"])
            ]
            label += [
                np.array(cur_dict[b"labels"])
            ]
        # Concat them
        data = np.concatenate(data)
        label = np.concatenate(label)

    elif data_type == "valid":
        # We'll use the 5th batch as our validation data. Note that this is not
        # the best way to do this. One strategy I've seen is to use this to
        # figure out the loss value you should aim to train for, and then stop
        # at that point, using the entire dataset. However, for simplicity,
        # we'll use this simple strategy.
        data = []
        label = []
        cur_dict = unpickle(os.path.join(data_dir, "data_batch_5"))
        data = np.array(cur_dict[b"data"])
        label = np.array(cur_dict[b"labels"])

    elif data_type == "test":
        data = []
        label = []
        cur_dict = unpickle(os.path.join(data_dir, "test_batch"))
        data = np.array(cur_dict[b"data"])
        label = np.array(cur_dict[b"labels"])

    else:
        raise ValueError("Wrong data type {}".format(data_type))

    # Turn data into (NxCxHxW) format, so that we can easily process it, where
    # N=number of images, H=height, W=widht, C=channels. Note that this
    # is the default format for PyTorch.
    data = np.reshape(data, (-1, 3, 32, 32))

    return data, label


class CIFAR10Dataset(Dataset):
    def __init__(self, data_dir, mode):
        print("Loading CIFAR10 Dataset from {} for {}ing ...".format(data_dir, mode), end="")
        # Load data (note that we now simply load the raw data.
        data, label = load_data(data_dir, mode)
        self.data = data
        self.label = label
        self.sample_shp = data.shape[1:]
        print(" done.")

    def __len__(self):
        # return the number of elements at `self.data`
        return len(self.data)

    def __getitem__(self, index):
        # Grab one data from the dataset and then apply our feature extraction
        data_cur = self.data[index]
        # Make pytorch object
        data_cur = torch.from_numpy(data_cur.astype(np.float32))
        # Label is just the label
        label_cur = self.label[index]
        return data_cur, label_cur


class MyConv2d(nn.Module):

    def __init__(self, config, inchannel, outchannel, ksize, stride, padding, bias=False):
        super(MyConv2d, self).__init__()
        # Our custom convolution kernel. We'll initialize it using Kaiming He's
        # initialization with uniform distribution
        self.weight = nn.Parameter(
            torch.randn((outchannel, inchannel, ksize, ksize)),
            requires_grad=True)
        # self.bias = nn.Parameter(
        #     torch.randn((outchannel,)),
        #     requires_grad=True)
        self.ksize = ksize
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        assert(len(x.shape) == 4)
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
                cur_w = self.weight[:, :, _j, _i]
                # Left multiply
                cur_o = torch.matmul(cur_w, cur_x)
                # Return to original shape
                cur_o = cur_o.reshape(x.shape[0], self.weight.shape[0], h_n, w_n)
                # Cumulative sum
                if x_out is None:
                    x_out = cur_o
                else:
                    x_out += cur_o
        # # Add bias
        # x_out += self.bias.view(1, self.bias.shape[0], 1, 1)
        return x_out


class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, config, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = MyConv2d(config, in_planes, planes, ksize=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = MyConv2d(config, planes, planes, ksize=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class ResNet(nn.Module):

    def __init__(self, config, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.config = config
        self.conv1 = MyConv2d(config, inchannel=3, outchannel=64, ksize=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.config, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def model_loss(config, model):
    loss = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            loss += torch.sum(param**2)

    return loss


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    res = []
    with torch.no_grad():
        maxk = max(topk)
        batch_size = output.size(0)
        pred = output.topk(maxk, 1, largest=True, sorted=True)[1]
        correct = pred.eq(target.unsqueeze(dim=1).expand_as(pred)).t()
        for k in topk:
            correct_k = correct[:k].float().sum().mul_(100.0 / batch_size)
            res.append(correct_k)

    return res


if __name__ == "__main__":

    # ----------------------------------------
    # Parse configuration
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)
