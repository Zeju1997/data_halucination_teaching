from __future__ import absolute_import, division, print_function

# from trainer_blackbox_mixup_rl import Trainer
# from trainer_blackbox_implicit_cnn import Trainer
from trainer_blackbox_mixup_rl import Trainer
from options.options import Options
import os
import argparse
import sys

import yaml

sys.path.append('..') #Hack add ROOT DIR
from baseconfig import CONF

from types import SimpleNamespace

# the directory that options.py resides in
file_dir = os.path.dirname(__file__)

options = Options()
opts = options.parse()


# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONF.PATH.CONFIG, config_name)) as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    config = load_config("mnist_blackbox_unrolled.yaml")
    # config = load_config("moon.yaml")
    opts.set_defaults(**config)

    # generate data
    opts.init_data = True
    trainer = Trainer(opts.parse_args())
    trainer.main()
    # trainer.make_gif()
