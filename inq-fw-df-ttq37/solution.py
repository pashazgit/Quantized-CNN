
# Copyright (C), Visual Computing Group @ University of Victoria.
# Copyright (C), https://github.com/pytorch/examples/blob/master/imagenet/main.py
# Copyright (C), https://github.com/TropComplique/trained-ternary-quantization
# Copyright (C), https://github.com/Mxbonn/INQ-pytorch
# Copyright (C), https://github.com/vinsis/ternary-quantization


from pdb import set_trace as bp
import sys
sys.path.insert(0, "./inq-fw-df-ttq37/input_pipeline")
sys.path.insert(0, "./inq-fw-df-ttq37/inq")
sys.path.insert(0, "./inq-fw-df-ttq37/utils")
import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import get_config, print_usage
from tensorboardX import SummaryWriter
from input_pipeline.prepare_datasets import create_train_dataset, create_val_dataset
import torchvision
from inq.inqstep import inq_step
from inq.initialscales import initial_scales
from utils.getgrads import get_grads
from utils.ternarize import ternarize
from utils.optimizer import create_optimizer, copy_optimizer
from utils.accuracy import accuracy
from utils.reset_lr import reset_lr


def model_loss(model):
    loss = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            loss += torch.sum(param**2)

    return loss


def train_model(config, model, criterion, optimizer, optimizer_fp, optimizer_sf,
                train_loader, val_loader, train_writer, val_writer, iter_idx,
                scheduler, scheduler_fp, scheduler_sf, model_file_path):
    best_val_acc1 = 0

    # Make sure that the model is set for training
    model.train()
    # Training loop
    prefix = ''
    for epoch in tqdm(range(config.num_epoch), desc=prefix):
        # For each iteration
        for x, y in train_loader:
            # Counter
            iter_idx += 1
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
                # assert (not (group_sf['params'][0] < 0).sum())
                # assert (not (group_sf['params'][1] > 0).sum())
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

            # Validate results every validation interval
            if iter_idx % config.val_intv == 0:
                # Set model for evaluation
                model = model.eval()
                # List to contain all losses and accuracies for all the val batches
                val_loss = []
                val_acc1 = []
                val_acc5 = []
                for val_x, val_y in val_loader:
                    # Send data to GPU if we have one
                    if torch.cuda.is_available():
                        val_x, val_y = val_x.cuda(), val_y.cuda()
                    # Apply forward pass to compute the losses and accuracies for each of the val batches
                    with torch.no_grad():
                        # Compute logits
                        logits = model.forward(val_x)
                        # Compute loss and store as numpy
                        loss = criterion(logits, val_y) + config.weight_decay * model_loss(model)
                        val_loss.append(loss.cpu().numpy())
                        # Compute accuracy and store as numpy
                        acc1, acc5 = accuracy(logits, val_y, topk=(1, 5))
                        val_acc1.append(acc1.cpu().numpy())
                        val_acc5.append(acc5.cpu().numpy())
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
                model = model.train()
                if val_acc1_avg > best_val_acc1:
                    best_val_acc1 = val_acc1_avg
                    # Save best model using torch.save.
                    torch.save({
                        "model": model.state_dict(),
                    }, model_file_path)

        # scheduler.step()
        # scheduler_fp.step()
        # scheduler_sf.step()

    print("best_val_acc1: ", best_val_acc1)


def test_model(config):
    pass


def main(config):
    """The main function."""

    if config.mode == "train":
        # Create log directory and if it does not exist
        if not os.path.exists(config.log_dir):
            os.makedirs(config.log_dir)

        # Create save directory if it does not exist
        if not os.path.exists(config.save_dir):
            os.makedirs(config.save_dir)
        # Prepare model file path to save and load from
        model_file_path = os.path.join(config.save_dir, "model_inq_fw_df_ttq37.pth")

        # Initialize datasets for both training and validation
        train_dataset = create_train_dataset(config)
        val_dataset = create_val_dataset(config)

        # Create data loader for training and validation.
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=config.batch_size, num_workers=config.num_workers,
            shuffle=True, pin_memory=torch.cuda.is_available())
        val_loader = DataLoader(
            dataset=val_dataset, batch_size=config.batch_size, num_workers=config.num_workers,
            shuffle=False, pin_memory=torch.cuda.is_available())

        # Create model instance.
        model = torchvision.models.resnet101(pretrained=False)
        res = torch.load(os.path.join(config.model_pretrained, 'model_full_precision.pth'), map_location="cpu")
        model.load_state_dict(res['model'], strict=False)
        print('\nmodel created')
        # Move model to gpu if cuda is available
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()

        # Define the loss function (criterion)
        criterion = nn.CrossEntropyLoss()
        # Move criterion to gpu if cuda is available
        if torch.cuda.is_available():
            criterion = criterion.cuda()

        # create wrapped optimizer
        optimizer = create_optimizer(model, lr=config.learning_rate)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=list(map(float, config.milestones.strip('[]').split(','))))

        steps = list(map(float, config.quantiles.strip('[]').split(',')))  # incremental steps
        inq_idx = 0  # make counter start at zero

        prefix = ''
        for rnd in tqdm(range(len(steps)), desc=prefix):

            reset_lr(scheduler)

            # create optimizer for updating full precision version of targeted weights
            optimizer_fp = copy_optimizer(optimizer, lr=config.learning_rate)
            scheduler_fp = optim.lr_scheduler.MultiStepLR(optimizer_fp,
                                                          milestones=list(map(float, config.milestones.strip('[]').split(','))))
            if torch.cuda.is_available():
                assert (optimizer_fp.param_groups[0]['params'][0].is_cuda)

            # increment portion of targeted weights that supposed to be ternarized
            inq_step(optimizer_fp, steps, inq_idx)
            inq_idx += 1

            # initialize independent learnable scaling factors for each filter
            sf = initial_scales(optimizer_fp, config.threshold)

            # create optimizer for updating scaling factors
            optimizer_sf = optim.Adam(sf, lr=config.learning_rate_sf)
            scheduler_sf = optim.lr_scheduler.MultiStepLR(optimizer_sf,
                                                          milestones=list(map(float, config.milestones.strip('[]').split(','))))
            if torch.cuda.is_available():
                assert (optimizer_sf.param_groups[0]['params'][0].is_cuda)

            # initial tenarization
            ternarize(optimizer, optimizer_fp, optimizer_sf, config.threshold)

            # Create summary writer
            train_writer = SummaryWriter(
                log_dir=os.path.join(config.log_dir, "train_inq_fw_df_ttq37_{}".format(rnd)))
            val_writer = SummaryWriter(
                log_dir=os.path.join(config.log_dir, "valid_inq_fw_df_ttq37_{}".format(rnd)))

            iter_idx = -1  # make counter start at zero

            train_model(config, model, criterion, optimizer, optimizer_fp, optimizer_sf,
                        train_loader, val_loader, train_writer, val_writer, iter_idx,
                        scheduler, scheduler_fp, scheduler_sf, model_file_path)

    elif config.mode == "test":
        test_model(config)

    else:
        raise ValueError("Unknown run mode \"{}\"".format(config.mode))


if __name__ == "__main__":

    # ----------------------------------------
    # Parse configuration
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)
