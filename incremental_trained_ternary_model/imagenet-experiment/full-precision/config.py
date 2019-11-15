
# Copyright (C), Visual Computing Group @ University of Victoria.

import argparse


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
parser.add_argument("--hdf5_file_path", type=str,
                    default="/home/mostafa/Downloads/imagenet_trval_2012.h5",
                    help="path to dataset")

parser.add_argument("--learning_rate", type=float,
                    default=1e-4,
                    help="Learning rate (gradient step size)")

parser.add_argument("--batch_size", type=int,
                    default=128,
                    help="Size of each training batch")

parser.add_argument("--num_workers", type=int,
                    default=4,
                    help="How many subprocesses to use for data loading.")

parser.add_argument("--weight_decay", type=float,
                    default=1e-4,
                    help="L2 Regularization strength")


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


def print_usage():
    parser.print_usage()
