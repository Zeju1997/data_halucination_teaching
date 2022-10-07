from __future__ import absolute_import, division, print_function

# from trainer_blackbox_mixup_rl import Trainer
# from trainer_blackbox_implicit_cnn import Trainer
from trainer_optimized import Trainer

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


if __name__ == "__main__":
    options = Options()
    opts = options.parse()

    config = load_config("mnist.yaml")
    opts.set_defaults(**config)

    args = opts.parse_args()

    # generate data
    args.init_data = True

    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    trainer = Trainer(args)
    # trainer = Trainer(opts.parse_args())
    trainer.main()
    # trainer.make_gif()
    # trainer.plot_results()
    # trainer.plot_perceptual_loss()
