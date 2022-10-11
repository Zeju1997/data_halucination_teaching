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


def calc_results(opt, seeds, models, experiments):
    results = 'results_blackbox_implicit_{}.txt'.format(opt.data_mode)
    with open(results, 'a') as f:
        f.write('blackbox implicit final results ')
        f.writelines('\n')
    for experiment in experiments:
        for model in models:
            values = []
            for seed in seeds:
                model_name = "blackbox_implicit_" + opt.data_mode + "_" + str(opt.n_weight_update) + '_' + str(opt.n_z_update) + '_' + str(opt.epsilon)
                log_dir = os.path.join(CONF.PATH.LOG, model_name, str(seed), str(model), str(experiment))
                for file in os.listdir(log_dir):
                    if file.endswith(".csv"):
                        csv_file = os.path.join(log_dir, file)
                        with open(csv_file, "r") as file:
                            last_line = file.readlines()[-1]
                            last_line = last_line.strip('\n').split(',')
                            values.append(last_line[-1])
            values_np = np.asarray(values).astype(float)
            values_mean = np.mean(values_np)
            values_std = np.std(values_np)
            with open(results, 'a') as f:
                f.writelines('\n')
                f.write("Data: {}, Model: {}, Experiment: {}, Mean: {}, Std: {}".format(opt.data_mode, model, experiment, values_mean, values_std))
                f.writelines('\n')


seeds = [10094, 20058, 27026, 48495, 65800]

# config_file = ['mnist_blackbox_implicit.yaml', 'cifar10.yaml', 'cifar100.yaml']
# models = ['CNN3', 'CNN6', 'CNN9', 'CNN15']
# model = ['MLP']
experiments = ['SGD', 'Student', 'Baseline']


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

    parser.add_argument(
        '--collect', action='store_true', default=False,
        help='Collect results.')

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

    
    # trainer = Trainer(opts.parse_args())
    if not cli_args.collect:
        trainer = Trainer(args)
        trainer.main()
    else:
        print("Saving.")
        if cli_args.config == "mnist_blackbox_implicit.yaml":
            models = ['MLP']
        else:
            models = ['CNN3', 'CNN6', 'CNN9', 'CNN15']
        calc_results(args, seeds, models, experiments)
