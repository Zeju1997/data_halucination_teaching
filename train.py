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


seeds = [95873, 10094, 20058, 27026, 48495]

# config_file = ['mnist_blackbox_implicit.yaml', 'cifar10_blackbox_mixup.yaml']
models = ['CNN3', 'CNN6', 'CNN9', 'CNN15']
# models = ['MLP']
experiments = ['Student', 'SGD', 'Baseline']
# experiments = ['Student', 'Discrete_Mixup', 'Adam', 'Vanilla_Mixup']
# experiments = ['First_Order_Optimization', 'Second_Order_Optimization']

combination = list(itertools.product(seeds, models, experiments))


def calc_results(opt, seeds, models, experiments):
    results = 'results_blackbox_implicit_{}.txt'.format(opt.data_mode)
    # results = 'results_blackbox_mixup_rl_{}.txt'.format(opt.data_mode)
    with open(results, 'a') as f:
        # f.write('blackbox implicit final results ')
        f.write('blackbox mixup rl final results ')
        f.writelines('\n')
    for experiment in experiments:
        for model in models:
            values = []
            for seed in seeds:
                model_name = "blackbox_implicit_" + opt.data_mode + "_" + str(opt.n_weight_update) + '_' + str(opt.n_z_update) + '_' + str(opt.epsilon)
                # model_name = "blackbox_mixup_rl_" + opt.data_mode + "_" + str(opt.n_weight_update) + '_' + str(opt.n_z_update) + '_' + str(opt.epsilon)
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

    # trainer = Trainer(args)
    # trainer = Trainer(opts.parse_args())
    trainer.main()
    # trainer.make_gif()
    # trainer.plot_results()
    # trainer.plot_distribution()
    # trainer.plot_perceptual_loss()

    # calc_results(args, seeds, models, experiments)
