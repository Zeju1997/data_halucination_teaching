import os
from easydict import EasyDict

CONF = EasyDict()
CONF.PATH = EasyDict()

# Base Folder
CONF.PATH.BASE = os.path.abspath(os.path.dirname(__file__))
CONF.PATH.CONFIG = os.path.join(CONF.PATH.BASE, "configs")

# Data
CONF.PATH.DATA = os.path.join(CONF.PATH.BASE, "data")

# Outputs
CONF.PATH.OUTPUT = os.path.join(CONF.PATH.BASE, 'outputs')
CONF.PATH.LOG = os.path.join(CONF.PATH.BASE, 'log')
CONF.PATH.RESULTS = os.path.join(CONF.PATH.BASE, 'results')

