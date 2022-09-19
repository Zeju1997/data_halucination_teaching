import torch
from torchvision.utils import save_image, make_grid
import numpy as np
import matplotlib.pyplot as plt
import imageio
import glob
import os
import sys

import torchvision
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn

from sklearn.datasets import make_moons, make_classification
from sklearn.model_selection import train_test_split

# import utils

import csv

sys.path.append('..') #Hack add ROOT DIR
from baseconfig import CONF

visualize = False


def plot_classifier(model, max, min):
    w = 0
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            w = layer.state_dict()['weight'].cpu().numpy()

    slope = (-w[0, 0]/w[0, 1] - 1) / (1 + w[0, 1]/w[0, 0])

    x = np.linspace(min, max, 100)
    y = slope * x
    return x, y


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

        # sgd_example = utils.BaseConv(opt.eta)
        # tmp_student = utils.BaseConv(opt.eta)
        # baseline = utils.BaseConv(opt.eta)

    elif opt.data_mode == "mnist":
        print("Loading MNIST data ...")

        # MNIST normalizing
        # transform = transforms.Compose([transforms.ToTensor(),
        #                                 transforms.Normalize([0.5], [0.5])
        # ])

        transform = transforms.Compose([transforms.ToTensor(),
                                        lambda x: torch.round(x)])

        train_dataset = torchvision.datasets.MNIST(root=CONF.PATH.DATA, train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root=CONF.PATH.DATA, train=False, download=True, transform=transform)

        '''
        idx = (train_dataset.targets == opt.class_1) | (train_dataset.targets == opt.class_2)
        train_dataset.targets = train_dataset.targets[idx]
        train_dataset.data = train_dataset.data[idx]
        train_dataset.targets = np.where(train_dataset.targets == opt.class_1, 0, 1)
        indices = np.random.choice(len(train_dataset), opt.nb_train)
        train_dataset.data = train_dataset.data[indices]
        train_dataset.targets = train_dataset.targets[indices]
        
        idx = (test_dataset.targets == opt.class_1) | (test_dataset.targets == opt.class_2)
        test_dataset.targets = test_dataset.targets[idx]
        test_dataset.data = test_dataset.data[idx]
        test_dataset.targets = np.where(test_dataset.targets == opt.class_1, 0, 1)
        indices = np.random.choice(len(test_dataset), opt.nb_train)
        test_dataset.data = test_dataset.data[indices]
        test_dataset.targets = test_dataset.targets[indices]
        '''

        loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
        X = next(iter(loader))[0].numpy()
        Y = next(iter(loader))[1].numpy()

        # (N, W, H) = train_dataset.data.shape
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

    elif opt.data_mode == "gaussian":
        print("Generating Gaussian data ...")

        dim__diff = 7
        nb_data_per_class = 1000

        X, Y = init_data(opt.dim, nb_data_per_class)

        # sgd_example = utils.BaseLinear(opt.dim)
        # tmp_student = utils.BaseLinear(opt.dim)
        # baseline = utils.BaseLinear(opt.dim)

        if visualize:
            fig = plt.figure(figsize=(8, 5))
            a, b = plot_classifier(teacher, X.max(axis=0), X.min(axis=0))
            plt.plot(a, b, '-r', label='y=wx+b')
            plt.scatter(X[:, 0], X[:, 1], c=Y)
            plt.title('Gaussian Data')
            #plt.show()
            plt.close()

    elif opt.data_mode == "moon":
        print("Generating moon data ...")

        np.random.seed(0)
        noise_val = 0.2

        X, Y = make_moons(opt.nb_train+opt.nb_test, noise=noise_val)

        # sgd_example = utils.BaseLinear(opt.dim)
        # tmp_student = utils.BaseLinear(opt.dim)
        # baseline = utils.BaseLinear(opt.dim)

        if visualize:
            fig = plt.figure(figsize=(8, 5))
            a, b = plot_classifier(teacher, X.max(axis=0), X.min(axis=0))
            plt.plot(a, b, '-r', label='y=wx+b')
            plt.scatter(X[:, 0], X[:, 1], c=Y)
            plt.title('Moon Data')
            #plt.show()
            plt.close()

    elif opt.data_mode == "linearly_seperable":
        print("Generating linearly seperable data ...")

        X, Y = make_classification(
            n_samples=opt.nb_train+opt.nb_test, n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
        )
        rng = np.random.RandomState(2)
        X += 2 * rng.uniform(size=X.shape)

        # sgd_example = utils.BaseLinear(opt.dim)
        # tmp_student = utils.BaseLinear(opt.dim)
        # baseline = utils.BaseLinear(opt.dim)

        if visualize:
            fig = plt.figure(figsize=(8, 5))
            a, b = plot_classifier(teacher, X.max(axis=0), X.min(axis=0))
            plt.plot(a, b, '-r', label='y=wx+b')
            plt.scatter(X[:, 0], X[:, 1], c=Y)
            plt.title('Linearly Seperable Data')
            # plt.show()
            plt.close()
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


def plot_graphs(rootdir, experiment_dict, experiment_lst):
    mpl.rcParams['figure.dpi'] = 120
    mpl.rcParams['savefig.dpi'] = 200



    plt.figure()

    for experiment in experiment_lst:

        acc_np = 0
        w_diff_np = 0

        for i, file in tqdm(enumerate(sorted(experiment_dict[experiment]))):
            file_path = os.path.join(rootdir, file)
            if os.path.isfile(file_path):
                acc = []
                w_diff = []
                with open(file_path, 'r') as csvfile:
                    lines = csv.reader(csvfile, delimiter=',')
                    for idx, row in enumerate(lines):
                        if idx != 0:
                            acc.append(row[1])
                            w_diff.append(row[2])
                tmp_acc_np = np.asarray(acc).astype(float)
                tmp_w_diff_np = np.asarray(w_diff).astype(float)
                if i == 0:
                    acc_np = tmp_acc_np[np.newaxis, :]
                    w_diff_np = tmp_w_diff_np[np.newaxis, :]
                else:
                    acc_np = np.concatenate((acc_np, tmp_acc_np[np.newaxis, :]), axis=0)
                    w_diff_np = np.concatenate((w_diff_np, tmp_w_diff_np[np.newaxis, :]), axis=0)

        acc_mean = np.mean(acc_np, axis=0)
        acc_std = np.std(acc_np, axis=0)
        w_diff_mean = np.mean(w_diff_np, axis=0)
        w_diff_std = np.std(w_diff_np, axis=0)

        x = np.arange(len(acc_mean))
        plt.plot(x, acc_mean, label=r'acc', c='r')
        plt.fill_between(x, acc_mean-acc_std, acc_mean+acc_std, color=adjust_lightness('r', amount=0.5), alpha=0.3)

    plt.xlabel('latent_dim')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(r'DCI&HSIC')

    plt.savefig(os.path.join(rootdir, '{}_v5.png'.format(experiment)), bbox_inches='tight')
    sys.exit()
