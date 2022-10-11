from __future__ import absolute_import, division, print_function

# from trainer_blackbox_mixup_rl import Trainer
# from trainer_vae_mnist import Trainer
import itertools

from trainer_blackbox_implicit import Trainer

from options.options import Options
import os
import argparse
import sys

import yaml

sys.path.append('..') #Hack add ROOT DIR
from baseconfig import CONF

import numpy as np
import torch

import random

from argparse import ArgumentParser


# the directory that options.py resides in
file_dir = os.path.dirname(__file__)


# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONF.PATH.CONFIG, config_name)) as file:
        config = yaml.safe_load(file)
    return config


# seeds = [10094, 20058, 27026, 48495, 65800]

# config_file = ['mnist_blackbox_implicit.yaml', 'cifar10.yaml', 'cifar100.yaml']
# models = ['CNN3', 'CNN6', 'CNN9', 'CNN15']
# model = ['MLP']
# experiments = ['SGD', 'Student', 'Baseline']


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--config", default="",
        help="Config file name.")
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed.")
    parser.add_argument(
        "--model", choices=['CNN3', 'CNN6', 'CNN9', 'CNN15', 'MLP'],
        help="Model options.")
    parser.add_argument(
        "--experiment", choices=['SGD', 'Student', 'Baseline'],
        help="Experiment options.")

    cli_args = parser.parse_args()

    print(cli_args)

    options = Options()
    opts = options.parse()

    # config = load_config("mnist_blackbox_implicit.yaml")
    # config = load_config("cifar10.yaml")
    # config = load_config("cifar100.yaml")

    config = load_config(cli_args.config)
    opts.set_defaults(**config)

    args = opts.parse_args()

    # generate data
    args.init_data = True

    args.seed = cli_args.seed
    args.model = cli_args.model
    args.experiment = cli_args.experiment

    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    
    trainer = Trainer(args)
    # trainer = Trainer(opts.parse_args())
    trainer.main()
    # trainer.make_gif()
    # trainer.plot_results()
    # trainer.plot_distribution()
    # trainer.plot_perceptual_loss()
