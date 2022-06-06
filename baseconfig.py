import os
from easydict import EasyDict

CONF = EasyDict()
CONF.PATH = EasyDict()

# Base Folder
CONF.PATH.BASE = os.path.abspath(os.path.dirname(__file__)) #Base Graph3DMOT path
CONF.PATH.CONFIG = os.path.join(CONF.PATH.BASE, "configs")

# Data
CONF.PATH.DATA = os.path.join(CONF.PATH.BASE, "data")

# Outputs/ Logging
CONF.PATH.OUTPUT = os.path.join(CONF.PATH.BASE, 'outputs')

