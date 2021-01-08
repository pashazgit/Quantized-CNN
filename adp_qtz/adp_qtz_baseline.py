
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
                    default="/home/pasha/scratch/jobs/output/adp_qtz/baseline/saves",
                    help="Directory to save the best model")

parser.add_argument("--log_dir", type=str,
                    default="/home/pasha/scratch/jobs/output/adp_qtz/baseline/logs",
                    help="Directory to save logs and current model")

parser.add_argument("--learning_rate", type=float,
                    default=1e-1,
                    help="Learning rate (gradient step size)")

parser.add_argument("--batch_size", type=int,
                    default=128,
                    help="Size of each training batch")

parser.add_argument("--num_epoch", type=int,
                    default=400,
                    help="Number of epochs to train")

parser.add_argument("--val_intv", type=int,
                    default=350,
                    help="Validation interval")

parser.add_argument("--rep_intv", type=int,
                    default=350,
                    help="Report interval")

parser.add_argument("--l2_reg", type=float,
                    default=5e-4,
                    help="L2 Regularization strength")

parser.add_argument("--resume", type=str2bool,
                    default=True,
                    help="Whether to resume training from existing checkpoint")

parser.add_argument("--name_idx", type=int,
                    default=1,
                    help="")


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

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4821, 0.4462), (0.2472, 0.2437, 0.2617)),
    ])

    transform_valid = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4821, 0.4462), (0.2472, 0.2437, 0.2617)),
    ])

    # Initialize datasets for both training and validation
    train_data = CIFAR10Dataset(config.data_dir, mode="train", transform=transform_train)
    val_data = CIFAR10Dataset(config.data_dir, mode="valid", transform=transform_valid)

    # Create data loader for training and validation.
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=config.batch_size,
        num_workers=2,
        shuffle=True)
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=100,
        num_workers=2,
        shuffle=False)

    # Create model instance (ResNet20)
    model = ResNet(3)
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
    # optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)

    # Create log directory and save directory if it does not exist
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    # Create summary writer
    train_writer = SummaryWriter(
        log_dir=os.path.join(config.log_dir, "train_{}".format(config.name_idx)))
    val_writer = SummaryWriter(
        log_dir=os.path.join(config.log_dir, "valid_{}".format(config.name_idx)))

    # Initialize training
    start_epoch = 0
    iter_idx = -1  # make counter start at zero
    best_val_acc = 0  # to check if best validation accuracy
    # Prepare checkpoint file and model file to save and load from
    checkpoint_file = os.path.join(config.save_dir, "checkpoint_{}.path".format(config.name_idx))
    bestmodel_file = os.path.join(config.save_dir, "bestmodel_{}.path".format(config.name_idx))

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
            best_val_acc = load_res["best_val_acc"]
            # Resume model
            model.load_state_dict(load_res["model"])
            # Resume optimizer
            optimizer.load_state_dict(load_res["optimizer"])
        else:
            os.remove(checkpoint_file)

    # Training loop
    for epoch in tqdm(range(start_epoch, config.num_epoch)):
        if epoch == 81:
            optimizer.param_groups[0]['lr'] = 0.01
            print('learning_rate: ', optimizer.param_groups[0]['lr'])
        elif epoch ==122:
            optimizer.param_groups[0]['lr'] = 0.001
            print('learning_rate: ', optimizer.param_groups[0]['lr'])
        elif epoch == 299:
            optimizer.param_groups[0]['lr'] = 0.0002
            print('learning_rate: ', optimizer.param_groups[0]['lr'])

        for x, y in train_loader:
            iter_idx += 1  # Counter
            # Send data to GPU if we have one
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            # Apply the model to obtain scores (forward pass)
            logits = model.forward(x)
            # Compute the loss
            loss = criterion(logits, y) + model_loss(config, model)
            # Compute gradients
            loss.backward()
            optimizer.step()
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Monitor results every report interval
            if iter_idx % config.rep_intv == 0:
                # Compute accuracy (No gradients required). We'll wrapp this
                # part so that we prevent torch from computing gradients.
                with torch.no_grad():
                    pred = torch.argmax(logits, dim=1)
                    acc = torch.mean(torch.eq(pred, y).float()) * 100.0
                # Write loss and accuracy to tensorboard, using keywords `loss` and `accuracy`.
                train_writer.add_scalar("data/loss", loss, global_step=iter_idx)
                train_writer.add_scalar("data/accuracy", acc, global_step=iter_idx)
                # Save
                torch.save({
                    "epoch": epoch,
                    "iter_idx": iter_idx,
                    "best_val_acc": best_val_acc,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict()
                }, checkpoint_file)

            # Validate results every validation interval
            if iter_idx % config.val_intv == 0:
                # Set model for evaluation
                model.eval()
                # List to contain all losses and accuracies for all the val batches
                val_loss = []
                val_acc = []
                for x, y in val_loader:
                    # Send data to GPU if we have one
                    if torch.cuda.is_available():
                        x, y = x.cuda(), y.cuda()
                    # Apply forward pass to compute the losses and accuracies for each of the val batches
                    with torch.no_grad():
                        # Compute logits
                        logits = model.forward(x)
                        # Compute loss and store as numpy
                        loss = criterion(logits, y) + model_loss(config, model)
                        # print(torch.cuda.is_available())
                        val_loss += [loss.cpu().numpy()]
                        # Compute accuracy and store as numpy
                        pred = torch.argmax(logits, dim=1)
                        acc = torch.mean(torch.eq(pred, y).float()) * 100.0
                        val_acc += [acc.cpu().numpy()]
                # Take average
                val_loss_avg = np.mean(val_loss)
                val_acc_avg = np.mean(val_acc)
                # Write loss and accuracy to tensorboard, using keywords `loss` and `accuracy`.
                val_writer.add_scalar("data/loss", val_loss_avg, global_step=iter_idx)
                val_writer.add_scalar("data/accuracy", val_acc_avg, global_step=iter_idx)
                # Set model back for training
                model.train()
                if val_acc_avg > best_val_acc:
                    best_val_acc = val_acc_avg
                    # Save
                    torch.save({
                        "epoch": epoch,
                        "iter_idx": iter_idx,
                        "best_val_acc": best_val_acc,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict()
                    }, bestmodel_file)

    print("best_val_accuracy: ", best_val_acc)


def test(config):
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

    # Create model instance (ResNet20)
    model = ResNet(3)
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
    load_res = torch.load(
        bestmodel_file,
        map_location="cpu")
    model.load_state_dict(load_res["model"])

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
            # Compute loss and store as numpy
            loss = criterion(logits, y) + model_loss(config, model)
            # print(torch.cuda.is_available())
            test_loss += [loss.cpu().numpy()]
            # Compute accuracy and store as numpy
            pred = torch.argmax(logits, dim=1)
            acc = torch.mean(torch.eq(pred, y).float()) * 100.0
            test_acc += [acc.cpu().numpy()]
    # Take average
    test_loss_avg = np.mean(test_loss)
    test_acc_avg = np.mean(test_acc)

    # Report Test loss and accuracy
    print("test_loss: ", test_loss_avg)
    print("test_accuracy: ", test_acc_avg)


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

    def __init__(self, data_dir, mode, transform=None):
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


class Residual(nn.Module):

    def __init__(self, in_channel, increase_dim):
        super(Residual, self).__init__()
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


class PreResidual(nn.Module):

    def __init__(self, in_channel, increase_dim=False):
        super(PreResidual, self).__init__()
        out_channel = in_channel
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = F.relu(self.bn2(self.conv1(x)))
        out = (self.conv2(out))
        out += x
        return out


class ResNet(nn.Module):

    def __init__(self, n):
        super(ResNet, self).__init__()
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
            layers.append(Residual(in_channel=in_plane, increase_dim=True))
            in_plane *= 2
        else:
            layers.append(PreResidual(in_channel=in_plane, increase_dim=False))

        for k in range(1, n):
            layers.append(Residual(in_channel=in_plane, increase_dim=False))

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


def model_loss(config, model):
    loss = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            loss += torch.sum(param**2)

    return loss * config.l2_reg


if __name__ == "__main__":

    # ----------------------------------------
    # Parse configuration
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)
