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

# the directory that options.py resides in
file_dir = os.path.dirname(__file__)


# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONF.PATH.CONFIG, config_name)) as file:
        config = yaml.safe_load(file)
    return config


seeds = [10094, 20058, 27026, 48495, 51626, 57890, 65800, 70293]

# config_file = ['mnist_blackbox_implicit.yaml', 'cifar10.yaml', 'cifar100.yaml']
models = ['CNN3', 'CNN6', 'CNN9', 'CNN15']
# model = ['MLP']
experiments = ['SGD', 'Student', 'Baseline']

combination = list(itertools.product(seeds, models, experiments))

if __name__ == "__main__":
    options = Options()
    opts = options.parse()

    # config = load_config("mnist_blackbox_implicit.yaml")
    config = load_config("cifar10.yaml")
    # config = load_config("cifar100.yaml")

    opts.set_defaults(**config)

    args = opts.parse_args()

    curr_comb = combination[args.idx]

    # generate data
    args.init_data = True

    args.seed = curr_comb[0]
    args.model = curr_comb[1]
    args.experiment = curr_comb[2]

    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(curr_comb)

    sys.exit()

    trainer = Trainer(args)
    # trainer = Trainer(opts.parse_args())
    trainer.main()
    # trainer.make_gif()
    # trainer.plot_results()
    # trainer.plot_distribution()
    # trainer.plot_perceptual_loss()
