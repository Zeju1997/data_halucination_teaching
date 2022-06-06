import subprocess
import sys
import os
import random
import time
import shutil


# scene_names = [f for f in os.listdir(scene_dir)]
gta5_file = open(os.path.join(os.getcwd(), "splits/gta5/gta5_truncated.txt"), "r")
gta5s = gta5_file.read().split('\n')
gta5_data_dir = os.path.join(os.getcwd(), "datasets/gta5/images")
gta5_label_dir = os.path.join(os.getcwd(), "datasets/gta5/labels")
gta5_names = [f for f in os.listdir(gta5_data_dir) if f != ""]

i = 0
'''
for gta5 in gta5_names:
    if gta5 not in gta5s:
        data_dir = os.path.join(gta5_data_dir, gta5)
        try:
            i = i + 1
            os.remove(data_dir)
            print("remove scene", data_dir)
        except OSError as e:
            print("Error: %s : %s" % (data_dir, e.strerror))
'''

gta5_names = [f for f in os.listdir(gta5_label_dir) if f != ""]
for gta5 in gta5_names:
    if gta5 not in gta5s:
        label_dir = os.path.join(gta5_label_dir, gta5)
        try:
            i = i + 1
            os.remove(label_dir)
            print("remove scene", label_dir)
        except OSError as e:
            print("Error: %s : %s" % (label_dir, e.strerror))

print("In total: {} images deleted".format(i))
