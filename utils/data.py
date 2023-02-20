import sys

import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn

from sklearn.datasets import make_moons

from matplotlib.ticker import FormatStrFormatter


import csv

sys.path.append('..') #Hack add ROOT DIR
from baseconfig import CONF

visualize = False


def init_data(opt):
    if opt.data_mode == "cifar10":
        print("Loading CIFAR10 data ...")

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])

        train_dataset = torchvision.datasets.CIFAR10(root=CONF.PATH.DATA, train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root=CONF.PATH.DATA, train=False, download=True, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

        X = next(iter(train_loader))[0].numpy()
        Y = next(iter(train_loader))[1].numpy()
        (N, W, H, C) = train_dataset.data.shape
        dim = W*H*C

    elif opt.data_mode == "mnist":
        print("Loading MNIST data ...")

        # MNIST normalizing
        if opt.generator_type == "vae":
            transform = transforms.Compose([transforms.ToTensor(),
                                           lambda x: torch.round(x)])
        else:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize([0.5], [0.5])
                                            ])

        train_dataset = torchvision.datasets.MNIST(root=CONF.PATH.DATA, train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root=CONF.PATH.DATA, train=False, download=True, transform=transform)

        loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
        X = next(iter(loader))[0].numpy()
        Y = next(iter(loader))[1].numpy()

        (N, W, H) = train_dataset.data.shape
        # dim = W*H
        # X = X.reshape((N, dim))

        # create new data set with class 1 as 0 and class 2 as 1
        f = (Y == opt.class_1) | (Y == opt.class_2)
        X = X[f]
        Y = Y[f]
        Y = np.where(Y == opt.class_1, 0, 1)

        # Shuffle datasets
        randomize = np.arange(X.shape[0])
        np.random.shuffle(randomize)
        X = X[randomize]
        Y = Y[randomize]

        img_shape = (opt.channels, opt.img_size, opt.img_size)
        proj_matrix = torch.empty(int(np.prod(img_shape)), opt.dim).normal_(mean=0, std=0.1)

        torch.save(proj_matrix, 'proj_matrix.pt')

    elif opt.data_mode == "moon":
        print("Generating moon data ...")

        # np.random.seed(0)
        noise_val = 0.2

        X, Y = make_moons(opt.nb_train+opt.nb_test, noise=noise_val)

    else:
        print("Unrecognized data!")
        sys.exit()

    torch.save(X, 'X.pt')
    torch.save(Y, 'Y.pt')


def load_experiment_result(opt):
    """Write an event to the tensorboard events file
    """
    csv_path = os.path.join(opt.log_path, 'results' + '_' + opt.experiment + '_' + str(opt.seed) + '.csv')

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


def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


# from unittest import result
import torch
import pdb
import matplotlib.pyplot as plt
from glob import glob
import os
import sys
import numpy as np
# from liftoff import parse_opts
# from matplotlib.pyplot import cm
# from loss_capacity.functions import HSIC
import matplotlib as mpl
from tqdm import tqdm
import seaborn as sns


def plot_graphs(rootdir, experiment_dict, experiment_lst, model_name):
    # mpl.rcParams['figure.dpi'] = 120
    # mpl.rcParams['savefig.dpi'] = 200

    sns.set()
    sns.set_style('white')
    sns.set_theme(style="ticks")
    sns.set_context("paper", font_scale=3, rc={"lines.linewidth": 2.5})

    palette = list(iter(sns.mpl_palette("tab10", 8)))
    # Plot acc results

    plt.figure(figsize=(8, 6))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    for experiment in experiment_lst:
        acc_np = 0
        for i, file in tqdm(enumerate(sorted(experiment_dict[experiment]))):
            file_path = os.path.join(rootdir, file)
            if os.path.isfile(file_path):
                acc = []
                with open(file_path, 'r') as csvfile:
                    lines = csv.reader(csvfile, delimiter=',')
                    for idx, row in enumerate(lines):
                        if idx != 0:
                            acc.append(row[1])
                tmp_acc_np = np.asarray(acc).astype(float)
                if i == 0:
                    acc_np = tmp_acc_np[np.newaxis, :]
                else:
                    acc_np = np.concatenate((acc_np, tmp_acc_np[np.newaxis, :]), axis=0)

        acc_mean = np.mean(acc_np, axis=0)
        acc_std = np.std(acc_np, axis=0) * 0.2

        x = np.arange(len(acc_mean))

        if experiment == 'SGD':
            plt.plot(x, acc_mean, label='SGD', color=palette[0])
            plt.fill_between(x, acc_mean-acc_std, acc_mean+acc_std, color=palette[0], alpha=0.2)

        elif experiment == 'IMT_Baseline':
            plt.plot(x, acc_mean, label='IMT', color=palette[2])
            plt.fill_between(x, acc_mean-acc_std, acc_mean+acc_std, color=palette[2], alpha=0.2)

        else:
            plt.plot(x, acc_mean, label='DHT', color=palette[3])
            plt.fill_between(x, acc_mean-acc_std, acc_mean+acc_std, color=palette[3], alpha=0.2)

    plt.ylabel('Accuracy')
    plt.xlabel('Number of iterations')
    plt.legend(loc='best', fontsize=22)

    plt.savefig(os.path.join(rootdir, 'paper_results_acc_{}.pdf'.format(model_name)), bbox_inches='tight')

    plt.figure(figsize=(8, 6))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    for experiment in experiment_lst:

        w_diff_np = 0

        for i, file in tqdm(enumerate(sorted(experiment_dict[experiment]))):
            file_path = os.path.join(rootdir, file)
            if os.path.isfile(file_path):
                w_diff = []
                with open(file_path, 'r') as csvfile:
                    lines = csv.reader(csvfile, delimiter=',')
                    for idx, row in enumerate(lines):
                        if idx != 0:
                            w_diff.append(row[2])
                tmp_w_diff_np = np.asarray(w_diff).astype(float)
                if i == 0:
                    w_diff_np = tmp_w_diff_np[np.newaxis, :]
                else:
                    w_diff_np = np.concatenate((w_diff_np, tmp_w_diff_np[np.newaxis, :]), axis=0)

        w_diff_mean = np.mean(w_diff_np, axis=0)
        w_diff_std = np.std(w_diff_np, axis=0) * 0.2

        x = np.arange(len(w_diff_mean))

        if experiment == 'SGD':
            plt.plot(x, w_diff_mean, label='SGD', color=palette[0])
            # plt.fill_between(x, w_diff_mean-w_diff_std, w_diff_mean+w_diff_std, color=adjust_lightness('yellow', amount=0.3), alpha=0.1)
            plt.fill_between(x, w_diff_mean-w_diff_std, w_diff_mean+w_diff_std, color=palette[0], alpha=0.2)

        elif experiment == 'IMT_Baseline':
            plt.plot(x, w_diff_mean, label='IMT', color=palette[2])
            # plt.fill_between(x, w_diff_mean-w_diff_std, w_diff_mean+w_diff_std, color=adjust_lightness('orange', amount=0.3), alpha=0.1)
            plt.fill_between(x, w_diff_mean-w_diff_std, w_diff_mean+w_diff_std, color=palette[2], alpha=0.2)

        else:
            plt.plot(x, w_diff_mean, label='DHT', color=palette[3])
            # plt.fill_between(x, w_diff_mean-w_diff_std, w_diff_mean+w_diff_std, color=adjust_lightness('r', amount=0.3), alpha=0.1)
            plt.fill_between(x, w_diff_mean-w_diff_std, w_diff_mean+w_diff_std, color=palette[3], alpha=0.2)

    plt.ylabel('Difference between $w*$ and $w_t$')
    plt.xlabel('Number of iterations')
    plt.legend(loc='best', fontsize=22)

    plt.savefig(os.path.join(rootdir, 'paper_results_w_diff_{}.pdf'.format(model_name)), bbox_inches='tight')


def plot_graphs_optimized(rootdir, experiment_dict, experiment_lst, model_name):
    # mpl.rcParams['figure.dpi'] = 120
    # mpl.rcParams['savefig.dpi'] = 200

    sns.set()
    sns.set_style('white')
    sns.set_theme(style="ticks")
    sns.set_context("paper", font_scale=3, rc={"lines.linewidth": 2.5})

    palette = list(iter(sns.mpl_palette("tab10", 8)))
    # Plot acc results

    plt.figure(figsize=(8, 6))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    for experiment in experiment_lst:
        acc_np = 0
        for i, file in tqdm(enumerate(sorted(experiment_dict[experiment]))):
            file_path = os.path.join(rootdir, file)
            if os.path.isfile(file_path):
                acc = []
                with open(file_path, 'r') as csvfile:
                    lines = csv.reader(csvfile, delimiter=',')
                    for idx, row in enumerate(lines):
                        if idx != 0:
                            acc.append(row[1])
                tmp_acc_np = np.asarray(acc).astype(float)
                if i == 0:
                    acc_np = tmp_acc_np[np.newaxis, :]
                else:
                    acc_np = np.concatenate((acc_np, tmp_acc_np[np.newaxis, :]), axis=0)

        acc_mean = np.mean(acc_np, axis=0)
        acc_std = np.std(acc_np, axis=0) * 0.2

        x = np.arange(len(acc_mean))

        if experiment == 'SGD':
            plt.plot(x, acc_mean, label='SGD', color=palette[0])
            plt.fill_between(x, acc_mean-acc_std, acc_mean+acc_std, color=palette[0], alpha=0.2)

        elif experiment == 'IMT_Baseline':
            plt.plot(x, acc_mean, label='IMT', color=palette[2])
            plt.fill_between(x, acc_mean-acc_std, acc_mean+acc_std, color=palette[2], alpha=0.2)

        elif experiment == 'Label':
            plt.plot(x, acc_mean, label='SGD+Label (R=2)', color=palette[1])
            plt.fill_between(x, acc_mean-acc_std, acc_mean+acc_std, color=palette[1], alpha=0.2)

        elif experiment == 'IMT_Label':
            plt.plot(x, acc_mean, label='IMT+Label (R=2))', color=palette[4])
            plt.fill_between(x, acc_mean-acc_std, acc_mean+acc_std, color=palette[4], alpha=0.2)

        elif experiment == 'Student':
            plt.plot(x, acc_mean, label='DHT', color=palette[3])
            plt.fill_between(x, acc_mean-acc_std, acc_mean+acc_std, color=palette[3], alpha=0.2)

        elif experiment == 'Student_with_Label':
            plt.plot(x, acc_mean, label='DHT+Label (R=2)', color=palette[7])
            plt.fill_between(x, acc_mean-acc_std, acc_mean+acc_std, color=palette[7], alpha=0.2)

    plt.ylabel('Accuracy')
    plt.xlabel('Number of iterations')
    # plt.legend(loc='best', fontsize=22)

    plt.savefig(os.path.join(rootdir, 'paper_results_acc_{}.pdf'.format(model_name)), bbox_inches='tight')

    # Plot w diff results
    plt.figure(figsize=(8, 6))
    plt.rcParams["font.family"] = "Times New Roman"

    for experiment in experiment_lst:

        w_diff_np = 0

        for i, file in tqdm(enumerate(sorted(experiment_dict[experiment]))):
            file_path = os.path.join(rootdir, file)
            if os.path.isfile(file_path):
                w_diff = []
                with open(file_path, 'r') as csvfile:
                    lines = csv.reader(csvfile, delimiter=',')
                    for idx, row in enumerate(lines):
                        if idx != 0:
                            w_diff.append(row[2])
                tmp_w_diff_np = np.asarray(w_diff).astype(float)
                if i == 0:
                    w_diff_np = tmp_w_diff_np[np.newaxis, :]
                else:
                    w_diff_np = np.concatenate((w_diff_np, tmp_w_diff_np[np.newaxis, :]), axis=0)

        w_diff_mean = np.mean(w_diff_np, axis=0)
        w_diff_std = np.std(w_diff_np, axis=0) * 0.2

        x = np.arange(len(w_diff_mean))

        if experiment == 'SGD':
            plt.plot(x, w_diff_mean, label='SGD', color=palette[0])
            plt.fill_between(x, w_diff_mean-w_diff_std, w_diff_mean+w_diff_std, color=palette[0], alpha=0.2)

        elif experiment == 'IMT_Baseline':
            plt.plot(x, w_diff_mean, label='IMT', color=palette[2])
            plt.fill_between(x, w_diff_mean-w_diff_std, w_diff_mean+w_diff_std, color=palette[2], alpha=0.2)

        elif experiment == 'Label':
            plt.plot(x, w_diff_mean, label='SGD+Label (R=2)', color=palette[1])
            plt.fill_between(x, w_diff_mean-w_diff_std, w_diff_mean+w_diff_std, color=palette[1], alpha=0.2)

        elif experiment == 'IMT_Label':
            plt.plot(x, w_diff_mean, label='IMT+Label (R=2))', color=palette[4])
            plt.fill_between(x, w_diff_mean-w_diff_std, w_diff_mean+w_diff_std, color=palette[4], alpha=0.2)

        elif experiment == 'Student':
            plt.plot(x, w_diff_mean, label='DHT', color=palette[3])
            plt.fill_between(x, w_diff_mean-w_diff_std, w_diff_mean+w_diff_std, color=palette[3], alpha=0.2)

        elif experiment == 'Student_with_Label':
            plt.plot(x, w_diff_mean, label='DHT+Label (R=2)', color=palette[7])
            plt.fill_between(x, w_diff_mean-w_diff_std, w_diff_mean+w_diff_std, color=palette[7], alpha=0.2)

    plt.ylabel('Difference between $w*$ and $w_t$')
    plt.xlabel('Number of iterations')
    plt.legend(loc='best', fontsize=22)

    plt.savefig(os.path.join(rootdir, 'paper_results_w_diff_{}.pdf'.format(model_name)), bbox_inches='tight')


def plot_graphs_vae_cgan(rootdir, experiment_dict, experiment_lst, model_name):
    # mpl.rcParams['figure.dpi'] = 120
    # mpl.rcParams['savefig.dpi'] = 200

    sns.set()
    sns.set_style('white')
    sns.set_theme(style="ticks")
    sns.set_context("paper", font_scale=3, rc={"lines.linewidth": 2.5})

    palette = list(iter(sns.mpl_palette("tab10", 8)))
    # Plot acc results

    plt.figure(figsize=(8, 6))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    for experiment in experiment_lst:
        acc_np = 0
        for i, file in tqdm(enumerate(sorted(experiment_dict[experiment]))):
            file_path = os.path.join(rootdir, file)
            if os.path.isfile(file_path):
                acc = []
                with open(file_path, 'r') as csvfile:
                    lines = csv.reader(csvfile, delimiter=',')
                    for idx, row in enumerate(lines):
                        if idx != 0:
                            acc.append(row[1])
                tmp_acc_np = np.asarray(acc).astype(float)
                if i == 0:
                    acc_np = tmp_acc_np[np.newaxis, :]
                else:
                    acc_np = np.concatenate((acc_np, tmp_acc_np[np.newaxis, :]), axis=0)

        acc_mean = np.mean(acc_np, axis=0)
        acc_std = np.std(acc_np, axis=0) * 0.2

        x = np.arange(len(acc_mean))

        if experiment == 'SGD':
            plt.plot(x, acc_mean, label='SGD', color=palette[0])
            plt.fill_between(x, acc_mean-acc_std, acc_mean+acc_std, color=palette[0], alpha=0.2)

        elif experiment == 'IMT_Baseline':
            plt.plot(x, acc_mean, label='IMT', color=palette[2])
            plt.fill_between(x, acc_mean-acc_std, acc_mean+acc_std, color=palette[2], alpha=0.2)

        elif experiment == 'Student_vae':
            plt.plot(x, acc_mean, label='DHT (VAE)', color=palette[1])
            plt.fill_between(x, acc_mean-acc_std, acc_mean+acc_std, color=palette[1], alpha=0.2)

        else:
            plt.plot(x, acc_mean, label='DHT (GAN)', color=palette[6])
            plt.fill_between(x, acc_mean-acc_std, acc_mean+acc_std, color=palette[6], alpha=0.2)

    plt.ylabel('Accuracy')
    plt.xlabel('Number of iterations')
    plt.legend(loc='best', fontsize=22)

    plt.savefig(os.path.join(rootdir, 'paper_results_acc_{}.pdf'.format(model_name)), bbox_inches='tight')

    plt.figure(figsize=(8, 6))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    for experiment in experiment_lst:

        w_diff_np = 0

        for i, file in tqdm(enumerate(sorted(experiment_dict[experiment]))):
            file_path = os.path.join(rootdir, file)
            if os.path.isfile(file_path):
                w_diff = []
                with open(file_path, 'r') as csvfile:
                    lines = csv.reader(csvfile, delimiter=',')
                    for idx, row in enumerate(lines):
                        if idx != 0:
                            w_diff.append(row[2])
                tmp_w_diff_np = np.asarray(w_diff).astype(float)
                if i == 0:
                    w_diff_np = tmp_w_diff_np[np.newaxis, :]
                else:
                    w_diff_np = np.concatenate((w_diff_np, tmp_w_diff_np[np.newaxis, :]), axis=0)

        w_diff_mean = np.mean(w_diff_np, axis=0)
        w_diff_std = np.std(w_diff_np, axis=0) * 0.2

        x = np.arange(len(w_diff_mean))

        if experiment == 'SGD':
            plt.plot(x, w_diff_mean, label='SGD', color=palette[0])
            # plt.fill_between(x, w_diff_mean-w_diff_std, w_diff_mean+w_diff_std, color=adjust_lightness('yellow', amount=0.3), alpha=0.1)
            plt.fill_between(x, w_diff_mean-w_diff_std, w_diff_mean+w_diff_std, color=palette[0], alpha=0.2)

        elif experiment == 'IMT_Baseline':
            plt.plot(x, w_diff_mean, label='IMT', color=palette[2])
            # plt.fill_between(x, w_diff_mean-w_diff_std, w_diff_mean+w_diff_std, color=adjust_lightness('orange', amount=0.3), alpha=0.1)
            plt.fill_between(x, w_diff_mean-w_diff_std, w_diff_mean+w_diff_std, color=palette[2], alpha=0.2)

        elif experiment == 'Student_vae':
            plt.plot(x, w_diff_mean, label='DHT (VAE)', color=palette[1])
            # plt.fill_between(x, w_diff_mean-w_diff_std, w_diff_mean+w_diff_std, color=adjust_lightness('r', amount=0.3), alpha=0.1)
            plt.fill_between(x, w_diff_mean-w_diff_std, w_diff_mean+w_diff_std, color=palette[1], alpha=0.2)

        else:
            plt.plot(x, w_diff_mean, label='DHT (GAN)', color=palette[6])
            # plt.fill_between(x, w_diff_mean-w_diff_std, w_diff_mean+w_diff_std, color=adjust_lightness('r', amount=0.3), alpha=0.1)
            plt.fill_between(x, w_diff_mean-w_diff_std, w_diff_mean+w_diff_std, color=palette[6], alpha=0.2)

    plt.ylabel('Difference between $w*$ and $w_t$')
    plt.xlabel('Number of iterations')
    plt.legend(loc='best', fontsize=22)

    plt.savefig(os.path.join(rootdir, 'paper_results_w_diff_{}.pdf'.format(model_name)), bbox_inches='tight')



def plot_perceptual_loss(log_path, experiments_lst, model_name, epsilon=50):

    rootdir = log_path

    file_lst = []

    for file in os.listdir(rootdir):
        if file.endswith('.csv'):
            if 'PERCEPTUAL' in file:
                file_lst.append(file)

    sns.set()
    sns.set_style('white')
    sns.set_theme(style="ticks")
    sns.set_context("paper", font_scale=3, rc={"lines.linewidth": 2.5})

    palette = list(iter(sns.mpl_palette("tab10", 8)))
    # Plot acc results

    plt.figure(figsize=(8, 6))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    for idx, experiment in enumerate(experiments_lst):
        value_np = 0
        for i, file in tqdm(enumerate(sorted(file_lst))):
            file_path = os.path.join(rootdir, file)
            if os.path.isfile(file_path):
                value = []
                with open(file_path, 'r') as csvfile:
                    lines = csv.reader(csvfile, delimiter=',')
                    for n, row in enumerate(lines):
                        if n != 0:
                            value.append(row[idx + 1])
                tmp_value_np = np.asarray(value).astype(float)
                if i == 0:
                    value_np = tmp_value_np[np.newaxis, :]
                else:
                    value_np = np.concatenate((value_np, tmp_value_np[np.newaxis, :]), axis=0)

        value_mean = np.mean(value_np, axis=0)
        value_std = np.std(value_np, axis=0)

        x = np.arange(len(value_mean))

        if experiment == 'SGD':
            plt.plot(x, value_mean, label='SGD', color=palette[0])
            plt.fill_between(x, value_mean - value_std, value_mean + value_std, color=palette[0], alpha=0.2)

        elif experiment == 'IMT_Baseline':
            plt.plot(x, value_mean, label='IMT', color=palette[2])
            plt.fill_between(x, value_mean - value_std, value_mean + value_std, color=palette[2], alpha=0.2)

        else:
            plt.plot(x, value_mean, label='DHT', color=palette[3])
            plt.fill_between(x, value_mean - value_std, value_mean + value_std, color=palette[3], alpha=0.2)

    plt.axhline(y=epsilon, color='k', linestyle='dashed', label="$\epsilon = {}$".format(epsilon), linewidth=4)

    plt.ylabel('Perceptual Loss')
    plt.xlabel('Number of iterations')
    plt.legend(loc='best', fontsize=22)

    plt.savefig(os.path.join(rootdir, 'paper_results_perceptual_{}.pdf'.format(model_name)), bbox_inches='tight')


def initialize_weights(m):
  if isinstance(m, nn.Conv2d):
      nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
      if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.BatchNorm2d):
      if m.bias is not None:
          nn.init.constant_(m.weight.data, 1)
          nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.Linear):
      if m.bias is not None:
          nn.init.kaiming_uniform_(m.weight.data)
          nn.init.constant_(m.bias.data, 0)
