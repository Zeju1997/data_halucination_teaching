from __future__ import absolute_import, division, print_function

import os
import argparse
import sys
import csv
import itertools
import importlib
import yaml
import random
import torch
import numpy as np
from options.options import Options

sys.path.append('..') #Hack add ROOT DIR
from baseconfig import CONF

# the directory that current file resides in
file_dir = os.path.dirname(__file__)

# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONF.PATH.CONFIG, config_name)) as file:
        config = yaml.safe_load(file)
    return config


# seeds = [95873, 10094, 20058, 27026, 48495]
seeds = [95873]

# config_file = ['mnist_blackbox_implicit.yaml', 'cifar10_blackbox_mixup.yaml']
models = ['CNN3', 'CNN6', 'CNN9', 'CNN15']
# models = ['MLP']
experiments = ['Student', 'SGD', 'Baseline']
# experiments = ['Student', 'Discrete_Mixup', 'Adam', 'Vanilla_Mixup']
# experiments = ['First_Order_Optimization', 'Second_Order_Optimization']

combination = list(itertools.product(seeds, models, experiments))


def get_parser():
    parser = argparse.ArgumentParser(description="Command line parser.")
    parser.add_argument('--config', help="configuration file *.yml", type=str, required=True)
    parser.add_argument('--teaching_policy', help="choose one policy from the teaching policy directory", type=str, required=True)
    return parser


if __name__ == "__main__":
    # command line parser
    parser = get_parser()
    p = parser.parse_args()

    # base config parser
    options = Options()
    opts = options.parse()

    # specify the config file
    config = load_config(p.config + '.yaml')
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

    module_name = "teaching_policy." + args.teaching_policy
    module = importlib.import_module(module_name)

    trainer = module.Trainer(args)

    trainer.main()
