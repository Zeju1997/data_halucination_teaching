from __future__ import absolute_import, division, print_function

import os
import subprocess
import glob as glob

import sys

sys.path.append('..') #Hack add ROOT DIR
from baseconfig import CONF

img_dir = os.path.join(CONF.PATH.LOG, "moon_blackbox_final_5")

os.chdir(img_dir)
subprocess.call([
    'ffmpeg', '-framerate', '8', '-i', 'results_%01d.png', '-r', '30', '-pix_fmt', 'yuv420p',
    'video_name.mp4'
])

# for file_name in glob.glob("*.png"):
#    os.remove(file_name)
