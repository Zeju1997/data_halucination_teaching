from __future__ import absolute_import, division, print_function

import os
import subprocess
import glob as glob
import csv

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import imageio

import seaborn as sns

import numpy as np

import sys

from tqdm import tqdm

from utils.data import plot_graphs, plot_graphs_optimized, plot_graphs_vae_cgan, plot_perceptual_loss
from utils.visualize import plot_distribution

sys.path.append('..') #Hack add ROOT DIR
from baseconfig import CONF


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


def plot_results(log_path, experiments_lst, experiment_dict, model_name):
    rootdir = log_path

    for experiment in experiments_lst:
        for file in os.listdir(rootdir):
            if file.endswith('.csv'):
                if experiment in file:
                    experiment_dict[experiment].append(file)

    # plot_graphs(rootdir, experiment_dict, experiments_lst, model_name)
    # plot_graphs_optimized(rootdir, experiment_dict, experiments_lst, model_name)
    plot_graphs_vae_cgan(rootdir, experiment_dict, experiments_lst, model_name)


def load_experiment_result(log_path, experiment, seed):
    """If already trained before, load the experiment results from the corresponding .csv file.
    """
    csv_path = os.path.join(log_path,
                            'results' + '_' + experiment + '_' + str(seed) + '.csv')

    if os.path.isfile(csv_path):
        acc = []
        w_diff = []
        with open(csv_path, 'r') as csvfile:
            lines = csv.reader(csvfile, delimiter=',')
            for idx, row in enumerate(lines):
                if idx != 0:
                    acc.append(row[1])
                    w_diff.append(row[2])
        acc_np = np.asarray(acc).astype(float)
        w_diff_np = np.asarray(w_diff).astype(float)

    return acc_np, w_diff_np


def make_gif(log_path, data_mode):
    video_dir = os.path.join(log_path, "video")

    images = []
    for file_name in tqdm(sorted(os.listdir(video_dir))):
        if file_name.endswith('.png'):
            file_path = os.path.join(video_dir, file_name)
            images.append(imageio.imread(file_path))
    gif_path = os.path.join(video_dir, 'results_{}.gif'.format(data_mode))
    # imageio.mimsave(gif_path, images, fps=20)
    imageio.mimsave(gif_path, images, fps=20)


if __name__ == '__main__':
    seeds = [22442, 27026, 43852, 43886, 44597, 58431, 65800, 78957, 86239, 95873]
    # seeds = [17913, 25821, 27732, 31154, 32367, 44112, 57595, 58431, 65800, 78957]
    experiments_lst = ['SGD', 'IMT_Baseline', 'Student_vae', 'Student_cgan']

    experiment_dict = {
        'SGD': [],
        'IMT_Baseline': [],
        'Student_vae': [],
        'Student_cgan': []
    }

    data_mode = 'moon'

    model_name = "omniscient_vae_cgan_" + data_mode

    log_path = os.path.join(CONF.PATH.RESULTS, model_name)

    plot_results(log_path, experiments_lst, experiment_dict, model_name)
    # plot_perceptual_loss(log_path, experiments_lst, experiment_dict, model_name)
