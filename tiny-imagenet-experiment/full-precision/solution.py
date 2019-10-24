
# Copyright (C), Visual Computing Group @ University of Victoria.
# Copyright (C), https://github.com/pytorch/examples/blob/master/imagenet/main.py
# Copyright (C), https://github.com/TropComplique/trained-ternary-quantization
# Copyright (C), https://github.com/Mxbonn/INQ-pytorch
# Copyright (C), https://github.com/vinsis/ternary-quantization


from pdb import set_trace as bp
import sys
sys.path.insert(0, "./full-precision/input_pipeline")
sys.path.insert(0, "./full-precision/utils")
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
from utils.accuracy import accuracy


def model_loss(model):
    loss = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            loss += torch.sum(param**2)

    return loss


def train_model(config, model, criterion, optimizer,
                train_loader, val_loader, train_writer, val_writer, iter_idx,
                model_file_path):
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
            # Compute gradients
            loss.backward()

            optimizer.step()

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
                if val_acc1_avg > best_val_acc1:
                    best_val_acc1 = val_acc1_avg
                    # Save best model using torch.save.
                    torch.save({
                        "model": model.state_dict(),
                    }, model_file_path)

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
        model_file_path = os.path.join(config.save_dir, "model_full_precision.pth")

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
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        # Create summary writer
        train_writer = SummaryWriter(
            log_dir=os.path.join(config.log_dir, "train_full_precision"))
        val_writer = SummaryWriter(
            log_dir=os.path.join(config.log_dir, "valid_full_precision"))

        iter_idx = -1  # make counter start at zero

        train_model(config, model, criterion, optimizer,
                    train_loader, val_loader, train_writer, val_writer, iter_idx,
                    model_file_path)

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
