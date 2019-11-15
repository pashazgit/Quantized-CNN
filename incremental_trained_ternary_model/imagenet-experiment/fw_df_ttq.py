4
# Copyright (C), Visual Computing Group @ University of Victoria.
# Copyright (C), https://github.com/pytorch/examples/blob/master/imagenet/main.py
# Copyright (C), https://github.com/TropComplique/trained-ternary-quantization
# Copyright (C), https://github.com/Mxbonn/INQ-pytorch
# Copyright (C), https://github.com/vinsis/ternary-quantization


from pdb import set_trace as bp
import sys
# sys.path.insert(0, "./fw-df-ttq41/input_pipeline")
# sys.path.insert(0, "./fw-df-ttq41/utils")
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

parser.add_argument("--hdf5_file_path", type=str,
                    default="/home/pasha/scratch/datasets/imagenet_trva_2012.h5",
                    help="")

parser.add_argument("--save_dir", type=str,
                    default="/home/pasha/scratch/jobs/output",
                    help="Directory to save the best model")

parser.add_argument("--log_dir", type=str,
                    default="/home/pasha/scratch/jobs/output",
                    help="Directory to save logs and current model")

parser.add_argument("--model_pretrained", type=str,
                    default="/home/pasha/scratch/imagenet_experiment/model_pretrained",
                    help="Directory to load the full precision pretrained model")

parser.add_argument("--learning_rate", type=float,
                    default=2e-5,
                    help="Learning rate (gradient step size)")

parser.add_argument("--learning_rate_sf", type=float,
                    default=1e-7,
                    help="Learning rate (gradient step size)")

parser.add_argument("--milestones", type=str,
                    default="[7]",
                    help="incremental steps")

parser.add_argument("--batch_size", type=int,
                    default=128,
                    help="Size of each training batch")

parser.add_argument("--num_epoch", type=int,
                    default=1,
                    help="Number of epochs to train")

parser.add_argument("--val_intv", type=int,
                    default=1000,
                    help="Validation interval")

parser.add_argument("--rep_intv", type=int,
                    default=1000,
                    help="Report interval")

parser.add_argument("--weight_decay", type=float,
                    default=1e-4,
                    help="L2 Regularization strength")

parser.add_argument("--threshold", type=float,
                    default=0.05,
                    help="ternarization threshold")

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
    train_data = train_dataset(config.hdf5_file_path)
    val_data = val_dataset(config.hdf5_file_path)

    # Create data loader for training and validation.
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=config.batch_size,
        num_workers=2,
        shuffle=True,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=config.batch_size,
        num_workers=2,
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )

    # Create model instance.
    model = torchvision.models.resnet101(pretrained=False)
    model_res = torch.load(os.path.join(config.model_pretrained, 'model_pretrained.pth'), map_location="cpu")
    model.load_state_dict(model_res['model'], strict=False)
    print('\nmodel created')
    # Move model to gpu if cuda is available
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    model.train()

    # Define the loss function (criterion)
    criterion = nn.CrossEntropyLoss()
    # Move criterion to gpu if cuda is available
    if torch.cuda.is_available():
        criterion = criterion.cuda()

    # create optimizer
    optimizer = create_optimizer(model, lr=config.learning_rate)
    optimizer_fp = copy_optimizer(optimizer, lr=config.learning_rate)
    if torch.cuda.is_available():
        assert optimizer_fp.param_groups[0]['params'][0].is_cuda
    sf = initial_scales(optimizer_fp, config.threshold)
    optimizer_sf = optim.Adam(sf, lr=config.learning_rate_sf)
    if torch.cuda.is_available():
        assert optimizer_sf.param_groups[0]['params'][0].is_cuda

    # Create summary writer
    train_writer = SummaryWriter(
        log_dir=os.path.join(config.log_dir, "train_fw_df_ttq"))
    val_writer = SummaryWriter(
        log_dir=os.path.join(config.log_dir, "valid_fw_df_ttq"))

    # Create log directory and save directory if it does not exist
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    # Initialize training
    start_epoch = 0
    iter_idx = -1  # make counter start at zero
    best_val_acc1 = 0  # to check if best validation accuracy
    # Prepare checkpoint file and model file to save and load from
    checkpoint_file = os.path.join(config.save_dir, "checkpoint_fw_df_ttq.pth")
    bestmodel_file = os.path.join(config.save_dir, "bestmodel_fw_df_ttq.pth")

    # initial tenarization
    ternarize(optimizer, optimizer_fp, optimizer_sf, config.threshold)

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
                map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            start_epoch = load_res["epoch"]
            # Resume iterations
            iter_idx = load_res["iter_idx"]
            # Resume best va result
            best_val_acc1 = load_res["best_val_acc1"]
            # Resume model
            model.load_state_dict(load_res["model"])
            # Resume optimizer
            optimizer.load_state_dict(load_res["optimizer"])
            optimizer_fp.load_state_dict(load_res["optimizer_fp"])
            optimizer_sf.load_state_dict(load_res["optimizer_sf"])
        else:
            os.remove(checkpoint_file)

    # Training loop
    for epoch in tqdm(range(start_epoch, config.num_epoch)):
        print("fw_sf_ttq_epoch: ", epoch)
        for data in train_loader:
            print("fw_sf_ttq_iter_idx: ", iter_idx)
            iter_idx += 1  # Counter
            x, y = data
            # Send data to GPU if we have one
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            # Apply the model to obtain scores (forward pass)
            logits = model.forward(x)
            # Compute the loss
            loss = criterion(logits, y) + config.weight_decay * model_loss(model)
            # Zero the parameter gradients
            optimizer.zero_grad()
            optimizer_fp.zero_grad()
            optimizer_sf.zero_grad()
            # Compute gradients
            loss.backward()
            # get modified grads and train
            get_grads(optimizer, optimizer_fp, optimizer_sf, config.threshold)
            optimizer.step()
            optimizer_fp.step()
            optimizer_sf.step()
            for group_sf in optimizer_sf.param_groups:
                try:
                    assert (not (group_sf['params'][0] < 0).sum())
                except AssertionError:
                    # a leaf Variable that requires grad cannot be used in an in-place operation.
                    group_sf['params'][0].data[group_sf['params'][0] < 0] = 0
                    assert (not (group_sf['params'][0] < 0).sum())
                try:
                    assert (not (group_sf['params'][1] > 0).sum())
                except AssertionError:
                    # a leaf Variable that requires grad cannot be used in an in-place operation.
                    group_sf['params'][1].data[group_sf['params'][1] > 0] = 0
                    assert (not (group_sf['params'][1] > 0).sum())
            ternarize(optimizer, optimizer_fp, optimizer_sf, config.threshold)
            # Monitor results every report interval
            if iter_idx % config.rep_intv == 0:
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
                    "optimizer": optimizer.state_dict(),
                    "optimizer_fp": optimizer_fp.state_dict(),
                    "optimizer_sf": optimizer_sf.state_dict()
                }, checkpoint_file)
            # Validate results every validation interval
            if iter_idx % config.val_intv == 0:
                # Set model for evaluation
                model = model.eval()
                # List to contain all losses and accuracies for all the val batches
                val_loss = []
                val_acc1 = []
                val_acc5 = []
                for data in val_loader:
                    x, y = data
                    # Send data to GPU if we have one
                    if torch.cuda.is_available():
                        x, y = x.cuda(), y.cuda()
                    # Apply forward pass to compute the losses and accuracies for each of the val batches
                    with torch.no_grad():
                        # Compute logits
                        logits = model.forward(x)
                        # Compute loss and store as numpy
                        loss = criterion(logits, y) + config.weight_decay * model_loss(model)
                        # print(torch.cuda.is_available())
                        val_loss += [loss.cpu().numpy]
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
                model = model.train()
                if val_acc1_avg > best_val_acc1:
                    best_val_acc1 = val_acc1_avg
                    # Save
                    torch.save({
                        "epoch": epoch,
                        "iter_idx": iter_idx,
                        "best_val_acc1": best_val_acc1,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "optimizer_fp": optimizer_fp.state_dict(),
                        "optimizer_sf": optimizer_sf.state_dict(),
                    }, bestmodel_file)

        print("fw_df_ttq_best_val_acc1: ", best_val_acc1)


def test(config):
    pass


class train_dataset(Dataset):
    def __init__(self, hdf5_file_path):
        self.hdf5_file = h5py.File(hdf5_file_path, mode='r')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             normalize])

    def __len__(self):
        return len(self.hdf5_file["train_images"])

    def __getitem__(self, idx):
        image = self.hdf5_file["train_images"][idx]
        image = self.transform(image)
        label = self.hdf5_file["train_labels"][idx]
        return image, label


class val_dataset(Dataset):
    def __init__(self, hdf5_file_path):
        self.hdf5_file = h5py.File(hdf5_file_path, mode='r')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             normalize])

    def __len__(self):
        return len(self.hdf5_file["val_images"])

    def __getitem__(self, idx):
        image = self.hdf5_file["val_images"][idx]
        image = self.transform(image)
        label = self.hdf5_file["val_labels"][idx]
        return image, label


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

    groups = [{'params': conv0_params, 'name': 'conv0_params'},
              {'params': conv1_params, 'name': 'conv1_params'},
              {'params': conv2_params, 'name': 'conv2_params'},
              {'params': conv3_params, 'name': 'conv3_params'},
              {'params': x_params, 'name': 'x_params'}]

    optimizer = optim.Adam(groups, lr=lr)

    return optimizer


def copy_optimizer(opt, lr=None):
    groups_fp = []
    for group in opt.param_groups[0:4]:
        # Notice: param.data.clone() is used instead of param.clone() because it can't optimize non-leaf Tensor
        a = torch.cat([param.data.clone().reshape(param.size(0), -1) for param in group['params']])
        groups_fp.append({'params': a})

    optimizer_fp = optim.Adam(groups_fp, lr=lr)

    return optimizer_fp


def initial_scales(opt_fp, thr):
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


def model_loss(model):
    loss = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            loss += torch.sum(param**2)

    return loss


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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = output.size(0)
        pred = output.topk(maxk, 1, largest=True, sorted=True)[1]
        correct = pred.eq(target.unsqueeze(dim=1).expand_as(pred)).t()
        res = []
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
