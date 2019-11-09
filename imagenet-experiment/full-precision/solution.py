
# Copyright (C), Visual Computing Group @ University of Victoria.
# Copyright (C), https://github.com/pytorch/examples/blob/master/imagenet/main.py
# Copyright (C), https://github.com/TropComplique/trained-ternary-quantization
# Copyright (C), https://github.com/Mxbonn/INQ-pytorch
# Copyright (C), https://github.com/vinsis/ternary-quantization


import sys
from pdb import set_trace as bp
sys.path.insert(0, "./full-precision/utils")
import os
import numpy as np
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from config import get_config, print_usage
import torchvision
from utils.accuracy import accuracy
from datawrapper import train_dataset, val_dataset
from torch.utils.data import DataLoader


def model_loss(model):
    loss = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            loss += torch.sum(param**2)

    return loss


def main(config):
    """The main function."""

    # Create model instance.
    model = torchvision.models.resnet101(pretrained=True)
    print('\nmodel created')
    # Move model to gpu if cuda is available
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    # Define the loss function (criterion)
    criterion = nn.CrossEntropyLoss()
    # Move criterion to gpu if cuda is available
    if torch.cuda.is_available():
        criterion = criterion.cuda()

    cudnn.benchmark = True

    valdataset = val_dataset(config.hdf5_file_path)

    val_loader = DataLoader(valdataset,
                            batch_size=config.batch_size, shuffle=False,
                            num_workers=config.workers, pin_memory=torch.cuda.is_available())

    model.eval()

    # List to contain all losses and accuracies for all the val batches
    test_loss = []
    test_acc1 = []
    test_acc5 = []
    for test_x, test_y in val_loader:
        # Send data to GPU if we have one
        if torch.cuda.is_available():
            test_x, test_y = test_x.cuda(), test_y.cuda()
        # Apply forward pass to compute the losses and accuracies for each of the val batches
        with torch.no_grad():
            # Compute logits
            logits = model.forward(test_x)
            # Compute loss and store as numpy
            loss = criterion(logits, test_y) + config.weight_decay * model_loss(model)
            test_loss.append(loss.cpu().numpy())
            # Compute accuracy and store as numpy
            acc1, acc5 = accuracy(logits, test_y, topk=(1, 5))
            test_acc1.append(acc1.cpu().numpy())
            test_acc5.append(acc5.cpu().numpy())
    # Take average
    test_loss_avg = np.mean(test_loss)
    test_acc1_avg = np.mean(test_acc1)
    test_acc5_avg = np.mean(test_acc5)

    # Report Test loss and accuracy
    print("Test Loss = {}".format(np.mean(test_loss_avg)))
    print("Test Accuracy = {}%".format(np.mean(test_acc1_avg)))


if __name__ == "__main__":

    # ----------------------------------------
    # Parse configuration
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)
