from __future__ import absolute_import, division, print_function

# from trainer_blackbox_mixup_rl import Trainer
# from trainer_vae_mnist import Trainer
import itertools

from trainer_blackbox_mixup_rl import Trainer

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

import csv

# the directory that options.py resides in
file_dir = os.path.dirname(__file__)


# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONF.PATH.CONFIG, config_name)) as file:
        config = yaml.safe_load(file)
    return config


seeds = [65800] #, 10094, 20058, 27026, 48495]

# config_file = ['mnist_blackbox_implicit.yaml', 'cifar10.yaml', 'cifar100.yaml']
models = ['CNN3'] # , 'CNN6', 'CNN9', 'CNN15']
# models = ['MLP']
experiments = ['SGD', 'Student', 'Baseline']

combination = list(itertools.product(seeds, models, experiments))

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
                f.write(" ".join(values))
                f.writelines('\n')


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

    trainer = Trainer(args)
    # trainer = Trainer(opts.parse_args())
    trainer.main()
    # trainer.make_gif()
    # trainer.plot_results()
    # trainer.plot_distribution()
    # trainer.plot_perceptual_loss()

    # calc_results(args, seeds, models, experiments)
