
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
parser.add_argument("--mode", type=str,
                    default="train",
                    choices=["train", "test"],
                    help="Run mode")

parser.add_argument("--data_dir", type=str,
                    default="./implementation/dataset/tiny-imagenet-200",
                    help="")

parser.add_argument("--save_dir", type=str,
                    default="./implementation/inq-fw-df-ttq37/saves",
                    help="Directory to save the best model")

parser.add_argument("--log_dir", type=str,
                    default="./implementation/inq-fw-df-ttq37/logs",
                    help="Directory to save logs and current model")

parser.add_argument("--model_pretrained", type=str,
                    default="./implementation/model-pretrained",
                    help="Directory to save the full precision pretrained model")

parser.add_argument("--learning_rate", type=float,
                    default=2e-5,
                    help="Learning rate (gradient step size)")

parser.add_argument("--learning_rate_sf", type=float,
                    default=1e-6,
                    help="Learning rate (gradient step size)")

parser.add_argument("--milestones", type=str,
                    default="[7]",
                    help="incremental steps")

parser.add_argument("--batch_size", type=int,
                    default=128,
                    help="Size of each training batch")

parser.add_argument("--num_workers", type=int,
                    default=4,
                    help="How many subprocesses to use for data loading.")

parser.add_argument("--num_epoch", type=int,
                    default=45,
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

parser.add_argument("--quantiles", type=str,
                    default="[0.5, 0.75, 1]",
                    help="incremental steps")

parser.add_argument("--strategy", type=str,
                    default="pruning",
                    help="incremental strategy")

parser.add_argument("--threshold", type=float,
                    default=0.1,
                    help="ternarization threshold")


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


def print_usage():
    parser.print_usage()
