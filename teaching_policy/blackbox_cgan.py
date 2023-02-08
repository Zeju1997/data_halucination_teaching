from __future__ import absolute_import, division, print_function

import numpy as np
import time
import json

import sys

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.nn as nn
import json
from tqdm import tqdm
import torchvision
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from train_utils import *

import teachers.omniscient_teacher as omniscient

import teachers.utils as utils
import matplotlib.pyplot as plt


from datasets import BaseDataset

import networks.blackbox_cgan as blackbox

from sklearn.datasets import make_moons, make_classification
from sklearn.model_selection import train_test_split

import subprocess
import glob

sys.path.append('..') #Hack add ROOT DIR
from baseconfig import CONF


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        # torch.nn.init.kaiming_uniform_(m.weight)
        # m.bias.data.fill_(0.01)

def plot_classifier(model, max, min):
    w = 0
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            w = layer.state_dict()['weight'].cpu().numpy()

    slope = (-w[0, 0]/w[0, 1] - 1) / (1 + w[0, 1]/w[0, 0])

    x = np.linspace(min, max, 100)
    y = slope * x
    return x, y


def approx_fprime(generator, f, epsilon, args=(), f0=None):
    """
    See ``approx_fprime``.  An optional initial function value arg is added.

    """

    xk = generator.linear.weight

    if f0 is None:
        f0 = f(*((xk,) + args))
    grad = np.zeros((xk.shape[0], xk.shape[1]), float)
    # grad = torch.zeros(len(xk),).cuda()
    ei = np.zeros((xk.shape[0], xk.shape[1],), float)
    # ei = torch.zeros(len(xk),).cuda()
    for j in range(xk.shape[0]):
        for k in range(xk.shape[1]):
            ei[j, k] = 1.0
            d = epsilon * ei
            d = torch.Tensor(d).cuda()
            grad[j, k] = (f(*((xk + d,) + args)) - f0) / d[j, k]
            ei[j, k] = 0.0
    return grad, f0


def to_matrix(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]


def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.shape[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :, :, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

mean = (0.1307,)
std = (0.3081,)
def plot_mixed_images(images):
    inv_normalize = transforms.Normalize(
                      mean= [-m/s for m, s in zip(mean, std)],
                      std= [1/s for s in std]
                      )
    inv_PIL = transforms.ToPILImage()
    fig = plt.figure(figsize=(16, 3))
    for i in range(1, len(images) + 1):
        image = images[i-1]
        ax = fig.add_subplot(1, 8, i)
        inv_tensor = inv_normalize(image).cpu()
        ax.imshow(inv_PIL(inv_tensor))
    plt.show()


class Trainer:
    def __init__(self, options):
        self.opt = options

        self.opt.model_name = "blackbox_cgan_" + self.opt.data_mode

        self.opt.log_path = os.path.join(CONF.PATH.LOG, self.opt.model_name)

        self.visualize = True

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.get_teacher_student()

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.opt.log_path, mode))

    def get_teacher_student(self):
        if self.opt.data_mode == "cifar10":
            self.teacher = omniscient.OmniscientConvTeacher(self.opt.eta)
            self.student = omniscient.OmniscientConvStudent(self.opt.eta)
        else: # mnist / gaussian / moon
            self.teacher = omniscient.OmniscientLinearTeacher(self.opt.dim)

            torch.save(self.teacher.state_dict(), 'teacher_w0.pth')
            # self.teacher.load_state_dict(torch.load('pretrained/teacher.pth'))

            self.student = omniscient.OmniscientLinearStudent(self.opt.dim)

            self.baseline = omniscient.OmniscientLinearStudent(self.opt.dim)

            # self.teacher = omniscient.TeacherClassifier(self.opt.dim)
            # self.student = omniscient.StudentClassifier(self.opt.dim)
        self.student.load_state_dict(self.teacher.state_dict())
        self.baseline.load_state_dict(self.teacher.state_dict())

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def init_data(self, dim, nb_data_per_class):
        """
        Création des données gaussien
        :param dim: la dimension des données
        :param nb_data_per_class: le nombre d'exemple par classe
        :return: un tuple (données, labels)
        """
        X1 = np.random.multivariate_normal([0.5] * dim, np.identity(dim), nb_data_per_class)
        y1 = np.ones((nb_data_per_class,))

        X2 = np.random.multivariate_normal([-0.5] * dim, np.identity(dim), nb_data_per_class)
        y2 = np.zeros((nb_data_per_class,))

        X = np.concatenate((X1, X2), axis=0)
        y = np.concatenate((y1, y2), axis=0)

        indices = np.indices((nb_data_per_class * 2,))
        np.random.shuffle(indices)

        X = X[indices]
        y = y[indices]
        return X.squeeze(0), y.squeeze(0)

    def sample_image(self, net_G, n_row, batches_done):
        """Saves a grid of generated digits ranging from 0 to n_classes"""
        # Sample noise
        z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (n_row ** 2, self.opt.latent_dim))))
        # Get labels ranging from 0 to n_classes for n rows
        labels = np.array([num for _ in range(n_row) for num in range(n_row)])
        labels = Variable(torch.cuda.LongTensor(labels))
        gen_imgs = net_G(z, labels)
        save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)

    def data_sampler(self, X, Y, i):
        i_min = i * self.opt.batch_size
        i_max = (i + 1) * self.opt.batch_size

        x = X[i_min:i_max].cuda()
        y = Y[i_min:i_max].cuda()

        return x, y

    def main(self):
        """Run a single epoch of training and validation
        """

        # torch.manual_seed(self.opt.seed)
        # np.random.seed(self.opt.seed)
        # torch.cuda.manual_seed(self.opt.seed)
        # torch.cuda.set_device(args.gpu)
        # cudnn.benchmark = True
        # cudnn.enabled=True

        print("Training")
        # self.set_train()

        if self.opt.data_mode == "cifar10":
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
            y = next(iter(train_loader))[1].numpy()
            (N, W, H, C) = train_dataset.data.shape
            dim = W*H*C

            sgd_example = utils.BaseConv(self.opt.eta)
            tmp_student = utils.BaseConv(self.opt.eta)
            # baseline = utils.BaseConv(self.opt.eta)

        elif self.opt.data_mode == "mnist":
            print("Loading MNIST data ...")

            # MNIST normalizing
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize([0.5], [0.5])
            ])

            train_dataset = torchvision.datasets.MNIST(root=CONF.PATH.DATA, train=True, download=True, transform=transform)
            test_dataset = torchvision.datasets.MNIST(root=CONF.PATH.DATA, train=False, download=True, transform=transform)

            '''
            idx = (train_dataset.targets == self.opt.class_1) | (train_dataset.targets == self.opt.class_2)
            train_dataset.targets = train_dataset.targets[idx]
            train_dataset.data = train_dataset.data[idx]
            train_dataset.targets = np.where(train_dataset.targets == self.opt.class_1, 0, 1)
            indices = np.random.choice(len(train_dataset), self.opt.nb_train)
            train_dataset.data = train_dataset.data[indices]
            train_dataset.targets = train_dataset.targets[indices]
            
            idx = (test_dataset.targets == self.opt.class_1) | (test_dataset.targets == self.opt.class_2)
            test_dataset.targets = test_dataset.targets[idx]
            test_dataset.data = test_dataset.data[idx]
            test_dataset.targets = np.where(test_dataset.targets == self.opt.class_1, 0, 1)
            indices = np.random.choice(len(test_dataset), self.opt.nb_train)
            test_dataset.data = test_dataset.data[indices]
            test_dataset.targets = test_dataset.targets[indices]
            '''

            loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
            X = next(iter(loader))[0].numpy()
            Y = next(iter(loader))[1].numpy()

            sgd_example = utils.BaseLinear(self.opt.dim)

            # create new data set with class 1 as 0 and class 2 as 1
            f = (Y == self.opt.class_1) | (Y == self.opt.class_2)
            X = X[f]
            Y = Y[f]
            Y = np.where(Y == self.opt.class_1, 0, 1)

            # Shuffle datasets
            randomize = np.arange(X.shape[0])
            np.random.shuffle(randomize)
            X = X[randomize]
            Y = Y[randomize]

            sgd_example = utils.BaseLinear(self.opt.dim)
            tmp_student = utils.BaseLinear(self.opt.dim)

        elif self.opt.data_mode == "gaussian":
            print("Generating Gaussian data ...")

            dim__diff = 7
            nb_data_per_class = 1000

            X, Y = self.init_data(self.opt.dim, nb_data_per_class)

            sgd_example = utils.BaseLinear(self.opt.dim)
            tmp_student = utils.BaseLinear(self.opt.dim)
            # baseline = utils.BaseLinear(self.opt.dim)

            if self.visualize:
                fig = plt.figure(figsize=(8, 5))
                a, b = plot_classifier(self.teacher, X.max(axis=0), X.min(axis=0))
                plt.plot(a, b, '-r', label='y=wx+b')
                plt.scatter(X[:, 0], X[:, 1], c=Y)
                plt.title('Gaussian Data')
                #plt.show()
                plt.close()

        elif self.opt.data_mode == "moon":
            print("Generating moon data ...")

            np.random.seed(0)
            noise_val = 0.2

            X, Y = make_moons(self.opt.nb_train+self.opt.nb_test, noise=noise_val)

            sgd_example = utils.BaseLinear(self.opt.dim)
            tmp_student = utils.BaseLinear(self.opt.dim)
            # baseline = utils.BaseLinear(self.opt.dim)

            if self.visualize:
                fig = plt.figure(figsize=(8, 5))
                a, b = plot_classifier(self.teacher, X.max(axis=0), X.min(axis=0))
                plt.plot(a, b, '-r', label='y=wx+b')
                plt.scatter(X[:, 0], X[:, 1], c=Y)
                plt.title('Moon Data')
                #plt.show()
                plt.close()

        elif self.opt.data_mode == "linearly_seperable":
            print("Generating linearly seperable data ...")

            X, Y = make_classification(
                n_samples=self.opt.nb_train+self.opt.nb_test, n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
            )
            rng = np.random.RandomState(2)
            X += 2 * rng.uniform(size=X.shape)

            sgd_example = utils.BaseLinear(self.opt.dim)
            tmp_student = utils.BaseLinear(self.opt.dim)
            # baseline = utils.BaseLinear(self.opt.dim)

            if self.visualize:
                fig = plt.figure(figsize=(8, 5))
                a, b = plot_classifier(self.teacher, X.max(axis=0), X.min(axis=0))
                plt.plot(a, b, '-r', label='y=wx+b')
                plt.scatter(X[:, 0], X[:, 1], c=Y)
                plt.title('Linearly Seperable Data')
                # plt.show()
                plt.close()
        else:
            print("Unrecognized data!")
            sys.exit()

        sgd_example.load_state_dict(self.teacher.state_dict())
        tmp_student.load_state_dict(self.teacher.state_dict())

        # Shuffle datasets
        # randomize = np.arange(X.shape[0])
        # np.random.shuffle(randomize)
        # X = X[randomize]
        # Y = Y[randomize]

        nb_batch = int(self.opt.nb_train / self.opt.batch_size)

        if self.opt.data_mode == "cifar10":
            X_train = torch.tensor(X[:self.opt.nb_train])
            Y_train = torch.tensor(Y[:self.opt.nb_train], dtype=torch.long)
            X_test = torch.tensor(X[self.opt.nb_train:self.opt.nb_train + self.opt.nb_test])
            Y_test = torch.tensor(Y[self.opt.nb_train:self.opt.nb_train + self.opt.nb_test], dtype=torch.long)

        elif self.opt.data_mode == "mnist":
            '''
            train_loader = DataLoader(train_dataset, batch_size=self.opt.batch_size, drop_last=True, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.opt.batch_size, drop_last=True, shuffle=False)

            X_train = train_dataset.data
            X_test = test_dataset.data
            Y_train = torch.tensor(train_dataset.targets, dtype=torch.float)
            Y_test = torch.tensor(test_dataset.targets, dtype=torch.float)

            X_train = X_train.view(X_train.shape[0], -1)
            X_test = X_test.view(X_test.shape[0], -1)

            img_shape = (self.opt.channels, self.opt.img_size, self.opt.img_size)
            proj_matrix = torch.empty(int(np.prod(img_shape)), self.opt.dim).normal_(mean=0, std=0.1)
            X_train = X_train.float() @ proj_matrix
            X_test = X_test.float() @ proj_matrix
            '''

            '''
            for i in range(50):
                tensor_image = X_test[i].squeeze()
                plt.imshow(tensor_image)
                print(Y_test[i])
                plt.show()

                print("aklsdfj")
            '''

            X_train = torch.tensor(X[:self.opt.nb_train], dtype=torch.float)
            Y_train = torch.tensor(Y[:self.opt.nb_train], dtype=torch.float)
            X_test = torch.tensor(X[self.opt.nb_train:self.opt.nb_train + self.opt.nb_test], dtype=torch.float)
            Y_test = torch.tensor(Y[self.opt.nb_train:self.opt.nb_train + self.opt.nb_test], dtype=torch.float)

            data_train = BaseDataset(X_train, Y_train)
            train_loader = DataLoader(data_train, batch_size=self.opt.batch_size, drop_last=True, shuffle=True)

            X_train = X_train.reshape((self.opt.nb_train, self.opt.img_size**2))
            X_test = X_test.reshape((self.opt.nb_test, self.opt.img_size**2))

            img_shape = (self.opt.channels, self.opt.img_size, self.opt.img_size)
            proj_matrix = torch.empty(int(np.prod(img_shape)), self.opt.dim).normal_(mean=0, std=0.1)
            X_train = X_train.float() @ proj_matrix
            X_test = X_test.float() @ proj_matrix

        else:
            X_train = torch.tensor(X[:self.opt.nb_train], dtype=torch.float)
            Y_train = torch.tensor(Y[:self.opt.nb_train], dtype=torch.float)
            X_test = torch.tensor(X[self.opt.nb_train:self.opt.nb_train + self.opt.nb_test], dtype=torch.float)
            Y_test = torch.tensor(Y[self.opt.nb_train:self.opt.nb_train + self.opt.nb_test], dtype=torch.float)

            proj_matrix = torch.eye(X.shape[1])

        # data_train = BaseDataset(X_train, Y_train)
        # data_test = BaseDataset(X_test, Y_test)
        # train_loader = DataLoader(data_train, batch_size=self.opt.batch_size, drop_last=True)
        # test_loader = DataLoader(data_test, batch_size=self.opt.batch_size, drop_last=True)

        # ---------------------
        #  Train Teacher
        # ---------------------

        accuracies = []
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.teacher.optim, milestones=[80, 160], gamma=0.1)
        for n in tqdm(range(self.opt.n_teacher_runs)):
            if n != 0:
                for i in range(nb_batch):
                    i_min = i * self.opt.batch_size
                    i_max = (i + 1) * self.opt.batch_size

                    self.teacher.update(X_train[i_min:i_max].cuda(), Y_train[i_min:i_max].cuda())

            self.teacher.eval()
            test = self.teacher(X_test.cuda()).cpu()

            if self.opt.data_mode == "mnist" or self.opt.data_mode == "gaussian" or self.opt.data_mode == "moon" or self.opt.data_mode == "linearly_seperable":
                tmp = torch.where(test > 0.5, torch.ones(1), torch.zeros(1))
                nb_correct = torch.where(tmp.view(-1) == Y_test, torch.ones(1), torch.zeros(1)).sum().item()
            elif self.opt.data_mode == "cifar10":
                tmp = torch.max(test, dim=1).indices
                nb_correct = torch.where(tmp == Y_test, torch.ones(1), torch.zeros(1)).sum().item()
            else:
                sys.exit()
            acc = nb_correct / X_test.size(0)
            accuracies.append(acc)
            print("Accuracy:", acc)
            # self.scheduler.step()

            if acc > 0.6 and n == 0:
                sys.exit()

        if self.visualize == True:
            fig = plt.figure()
            plt.plot(accuracies, c="b", label="Teacher (CNN)")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.close()

            # fig = plt.figure(figsize=(8, 5))
            # a, b = plot_classifier(self.teacher, X.max(axis=0), X.min(axis=0))
            # plt.plot(a, b, '-r', label='y=wx+b')
            # plt.scatter(X[:, 0], X[:, 1], c=Y)
            # plt.title('Initial Classifer Weight')
            # plt.close()

        w_star = self.teacher.lin.weight

        # ---------------------
        #  Train SGD
        # ---------------------

        res_sgd = []
        a_example = []
        b_example = []
        w_diff_sgd = []
        sgd_example.load_state_dict(torch.load('teacher_w0.pth'))
        for idx in tqdm(range(self.opt.n_iter)):
            sgd_example.train()
            if idx != 0:
                i = torch.randint(0, nb_batch, size=(1,)).item()
                i_min = i * self.opt.batch_size
                i_max = (i + 1) * self.opt.batch_size

                data = X_train[i_min:i_max].cuda()
                label = Y_train[i_min:i_max].cuda()

                random_data = data.detach().clone().cpu().numpy()
                random_label = label.detach().clone().cpu().numpy()
                if idx == 1:
                    random_samples = random_data # [np.newaxis, :]
                    random_labels = random_label # [np.newaxis, :]
                else:
                    random_samples = np.concatenate((random_samples, random_data), axis=0)
                    random_labels = np.concatenate((random_labels, random_label), axis=0)

                sgd_example.update(data, label)

            sgd_example.eval()
            test = sgd_example(X_test.cuda()).cpu()

            if self.opt.data_mode == "moon":
                a, b = plot_classifier(sgd_example, X.max(axis=0), X.min(axis=0))
                a_example.append(a)
                b_example.append(b)

            if self.opt.data_mode == "mnist" or self.opt.data_mode == "gaussian" or self.opt.data_mode == "moon":
                tmp = torch.where(test > 0.5, torch.ones(1), torch.zeros(1))
                nb_correct = torch.where(tmp.view(-1) == Y_test, torch.ones(1), torch.zeros(1)).sum().item()
            elif self.opt.data_mode == "cifar10":
                tmp = torch.max(test, dim=1).indices
                nb_correct = torch.where(tmp == Y_test, torch.ones(1), torch.zeros(1)).sum().item()
            else:
                sys.exit()

            acc = nb_correct / X_test.size(0)
            res_sgd.append(acc)

            diff = torch.linalg.norm(w_star - sgd_example.lin.weight, ord=2) ** 2
            w_diff_sgd.append(diff.detach().clone().cpu())

        # ---------------------
        #  Train IMT Baseline
        # ---------------------

        res_baseline = []
        a_baseline = []
        b_baseline = []
        w_diff_baseline = []
        self.baseline.load_state_dict(torch.load('teacher_w0.pth'))
        for t in tqdm(range(self.opt.n_iter)):
            self.baseline.train()
            if t != 0:
                i = self.teacher.select_example(self.baseline, X_train.cuda(), Y_train.cuda(), self.opt.batch_size)
                # i = torch.randint(0, nb_batch, size=(1,)).item()

                i_min = i * self.opt.batch_size
                i_max = (i + 1) * self.opt.batch_size

                best_data = X_train[i_min:i_max].cuda()
                best_label = Y_train[i_min:i_max].cuda()

                selected_data = best_data.detach().clone().cpu().numpy()
                selected_label = best_label.detach().clone().cpu().numpy()
                if t == 1:
                    selected_samples = selected_data # [np.newaxis, :]
                    selected_labels = selected_label # [np.newaxis, :]
                else:
                    selected_samples = np.concatenate((selected_samples, selected_data), axis=0)
                    selected_labels = np.concatenate((selected_labels, selected_label), axis=0)

                self.baseline.update(best_data, best_label)

            self.baseline.eval()
            test = self.baseline(X_test.cuda()).cpu()

            if self.opt.data_mode == "moon":
                a, b = plot_classifier(self.baseline, X.max(axis=0), X.min(axis=0))
                a_baseline.append(a)
                b_baseline.append(b)

            if self.opt.data_mode == "mnist" or self.opt.data_mode == "gaussian" or self.opt.data_mode == "moon" or self.opt.data_mode == "linearly_seperable":
                tmp = torch.where(test > 0.5, torch.ones(1), torch.zeros(1))
                nb_correct = torch.where(tmp.view(-1) == Y_test, torch.ones(1), torch.zeros(1)).sum().item()
            elif self.opt.data_mode == "cifar10":
                tmp = torch.max(test, dim=1).indices
                nb_correct = torch.where(tmp == Y_test, torch.ones(1), torch.zeros(1)).sum().item()
            else:
                sys.exit()
            acc = nb_correct / X_test.size(0)
            res_baseline.append(acc)

            print("Accuracies", acc)

            diff = torch.linalg.norm(w_star - self.baseline.lin.weight, ord=2) ** 2
            w_diff_baseline.append(diff.detach().clone().cpu())

            sys.stdout.write("\r" + str(t) + "/" + str(self.opt.n_iter) + ", idx=" + str(i) + " " * 100)
            sys.stdout.flush()

        print("Base line trained\n")

        # ---------------------
        #  Train Student
        # ---------------------

        adversarial_loss = torch.nn.BCELoss()

        if self.opt.data_mode == "moon":
            netG = blackbox.Generator_moon(self.opt, self.teacher, tmp_student).cuda()
            netD = blackbox.Discriminator_moon(self.opt).cuda()
            unrolled_optimizer = blackbox.UnrolledBlackBoxOptimizer_moon(opt=self.opt, teacher=self.teacher, student=tmp_student, generator=netG, X=X_train.cuda(), Y=Y_train.cuda(), proj_matrix=proj_matrix)
        else:
            netG = blackbox.Generator(self.opt, self.teacher, tmp_student).cuda()
            netD = blackbox.Discriminator(self.opt).cuda()
            unrolled_optimizer = blackbox.UnrolledBlackBoxOptimizer(opt=self.opt, teacher=self.teacher, student=tmp_student, generator=netG, X=X_train.cuda(), Y=Y_train.cuda(), proj_matrix=proj_matrix)

        netG.apply(weights_init)
        netD.apply(weights_init)

        optimD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

        self.step = 0
        loss_student = []
        img_shape = (1, 28, 28)
        w_init = self.student.lin.weight

        cls = torch.arange(self.opt.n_classes)
        onehot = torch.zeros(self.opt.n_classes, self.opt.n_classes).scatter_(1, cls.view(self.opt.n_classes, 1), 1)
        # reshape labels to image size, with number of labels as channel
        fill = torch.zeros([self.opt.n_classes, self.opt.n_classes, self.opt.img_size, self.opt.img_size])
        for i in range(self.opt.n_classes):
            fill[i, i, :, :] = 1

        test_labels = torch.tensor([0, 1]*self.opt.n_classes).type(torch.LongTensor)
        test_labels_onehot = onehot[test_labels].cuda()

        real = torch.ones((self.opt.batch_size, 1)).cuda()
        fake = torch.zeros((self.opt.batch_size, 1)).cuda()

        # Fix noise for testing generator and visualization
        z_test = torch.randn(self.opt.n_classes**2, self.opt.latent_dim).cuda()

        # List of values, which will be used for plotting purpose
        D_losses = []
        G_losses = []
        Dx_values = []
        DGz_values = []

        # number of training steps done on discriminator
        step = 0
        for epoch in tqdm(range(self.opt.n_epochs)):

            epoch_D_losses = []
            epoch_G_losses = []
            epoch_Dx = []
            epoch_DGz = []
            # iterate through data loader generator object
            for images, labels in tqdm(train_loader):

                real_samples = images.cuda() # real_samples
                real_labels = labels.long()

                step += 1

                ############################
                # Update G network: maximize log(D(G(z)))
                ###########################

                # if Ksteps of Discriminator training are done, update generator
                '''
                netG.zero_grad()

                z_out = netD(generated_samples, generated_labels_fill)
                g_loss = adversarial_loss(z_out, real)

                w_t = netG.state_dict()
                gradients, generator_loss, G_loss, z_out = unrolled_optimizer(w_t, w_star, w_init, netD, generated_samples, generated_labels_fill, real, g_loss)
                loss_student.append(generator_loss.item())

                with torch.no_grad():
                    for p, g in zip(netG.parameters(), gradients):
                        p.grad = g

                # G_loss.backward()
                optimG.step()
                '''

                generated_labels = (torch.rand(self.opt.batch_size, 1)*2).type(torch.LongTensor).squeeze(1)
                generated_labels_onehot = onehot[generated_labels].cuda()
                generated_labels_fill = fill[generated_labels].cuda()

                netG.zero_grad()
                w_t = netG.state_dict()
                gradients, generator_loss, G_loss, z_out, generated_samples = unrolled_optimizer(w_t, w_star, w_init, netD, generated_labels, real)
                loss_student.append(generator_loss.item())

                with torch.no_grad():
                    for p, g in zip(netG.parameters(), gradients):
                        p.grad = g

                '''
                z = torch.randn(self.opt.batch_size, self.opt.latent_dim).cuda()


                generated_samples = netG(z, generated_labels_onehot)

                z_out = netD(generated_samples, generated_labels_fill)
                G_loss = adversarial_loss(z_out, real)
                '''

                epoch_DGz.append(z_out.mean().item())
                epoch_G_losses.append(G_loss.item())

                # G_loss.backward()
                optimG.step()

                ############################
                # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################

                real_labels_fill = fill[real_labels].cuda()
                real_preds = netD(real_samples, real_labels_fill)
                D_real_loss = adversarial_loss(real_preds, real)

                fake_preds = netD(generated_samples.detach(), generated_labels_fill)
                D_fake_loss = adversarial_loss(fake_preds, fake)
                D_loss = D_real_loss + D_fake_loss

                # save values for plots
                epoch_D_losses.append(D_loss.item())
                epoch_Dx.append(real_preds.mean().item())

                netD.zero_grad()
                D_loss.backward()
                optimD.step()

            else:
                # calculate average value for one epoch
                D_losses.append(sum(epoch_D_losses)/len(epoch_D_losses))
                G_losses.append(sum(epoch_G_losses)/len(epoch_G_losses))
                Dx_values.append(sum(epoch_Dx)/len(epoch_Dx))
                DGz_values.append(sum(epoch_DGz)/len(epoch_DGz))

                print(f" Epoch {epoch+1}/{self.opt.n_epochs} Discriminator Loss {D_losses[-1]:.3f} Generator Loss {G_losses[-1]:.3f}"
                     + f" D(x) {Dx_values[-1]:.3f} D(G(x)) {DGz_values[-1]:.3f}")

                res_student = []
                a_student = []
                b_student = []
                w_diff_student = []

                self.student.load_state_dict(torch.load('teacher_w0.pth'))
                netG.eval()
                generated_samples = np.zeros(2)

                self.opt.batch_size = 1

                for idx in tqdm(range(self.opt.n_iter)):
                    if idx != 0:
                        w_t = self.student.lin.weight

                        i = torch.randint(0, nb_batch, size=(1,)).item()
                        gt_x, gt_y = self.data_sampler(X_train, Y_train, i)
                        gt_y_onehot = onehot[gt_y.long()].cuda()

                        # z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, gt_x.shape)))
                        z = Variable(torch.randn((self.opt.batch_size, self.opt.latent_dim))).cuda()

                        # w = torch.cat((w_t, w_t-w_star), dim=1).repeat(self.opt.batch_size, 1)
                        w_t = w_t.repeat(self.opt.batch_size, 1)
                        x = torch.cat((w_t, z), dim=1)
                        generated_sample = netG(x, gt_y_onehot)

                        if self.opt.data_mode == "moon":
                            if idx == 1:
                                generated_samples = generated_sample.cpu().detach().numpy()  # [np.newaxis, :]
                                generated_labels = gt_y.cpu().detach().numpy()  # [np.newaxis, :]
                            else:
                                generated_samples = np.concatenate((generated_samples, generated_sample.cpu().detach().numpy()), axis=0)
                                generated_labels = np.concatenate((generated_labels, gt_y.cpu().detach().numpy()), axis=0)
                        else:
                            generated_sample = generated_sample.view(self.opt.batch_size, -1)
                            generated_sample = generated_sample @ proj_matrix.cuda()

                        self.student.update(generated_sample, gt_y.unsqueeze(1))

                    self.student.eval()
                    test = self.student(X_test.cuda()).cpu()

                    if self.opt.data_mode == "moon":
                        a, b = plot_classifier(self.student, X.max(axis=0), X.min(axis=0))
                        a_student.append(a)
                        b_student.append(b)

                    if self.opt.data_mode == "mnist" or self.opt.data_mode == "gaussian" or self.opt.data_mode == "moon":
                        tmp = torch.where(test > 0.5, torch.ones(1), torch.zeros(1))
                        nb_correct = torch.where(tmp.view(-1) == Y_test, torch.ones(1), torch.zeros(1)).sum().item()
                    elif self.opt.data_mode == "cifar10":
                        tmp = torch.max(test, dim=1).indices
                        nb_correct = torch.where(tmp == Y_test, torch.ones(1), torch.zeros(1)).sum().item()
                    else:
                        sys.exit()
                    acc = nb_correct / X_test.size(0)
                    res_student.append(acc)

                    # diff = self.teacher.lin.weight - example.lin.weight
                    diff = torch.linalg.norm(self.teacher.lin.weight - self.student.lin.weight, ord=2) ** 2
                    w_diff_student.append(diff.detach().clone().cpu())

                if self.opt.data_mode == "mnist":
                    save_folder = os.path.join(self.opt.log_path, "imgs")
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)

                    img_path = os.path.join(save_folder, "results_examples_{}.png".format(epoch))

                    netG.eval()
                    with torch.no_grad():
                        w_t = self.student.lin.weight
                        w_t = w_t.repeat(self.opt.n_classes**2, 1)
                        x = torch.cat((w_t, z_test), dim=1)
                        fake_test = netG(x, test_labels_onehot).cpu()
                        torchvision.utils.save_image(fake_test, img_path, nrow=10, padding=0, normalize=True)
                    netG.train()

                    # z = Variable(torch.randn((self.opt.n_classes**2, self.opt.latent_dim))).cuda()
                    # w = torch.cat((w_t, w_t-w_star), dim=1).repeat(self.opt.n_classes**2, 1)
                    # x = torch.cat((w, z), dim=1)
                    # test_samples = netG(z, test_labels_onehot)

                if self.opt.data_mode == "gaussian" or self.opt.data_mode == "moon":
                    self.make_results_img_2d(X, Y, generated_samples, generated_labels, res_sgd, res_baseline, res_student, w_diff_sgd, w_diff_baseline, w_diff_student, epoch)
                else:
                    self.make_results_img(res_sgd, res_baseline, res_student, w_diff_sgd, w_diff_baseline, w_diff_student, loss_student, G_losses, D_losses, epoch, proj_matrix)

                save_folder = os.path.join(self.opt.log_path, "models", "weights_{}".format(epoch))
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)

                save_path = os.path.join(save_folder, "netG_{}.pth".format("models", epoch))
                to_save = netG.state_dict()
                torch.save(to_save, save_path)

                save_path = os.path.join(save_folder, "netD_{}.pth".format("models", epoch))
                to_save = netD.state_dict()
                torch.save(to_save, save_path)

        plt.figure(figsize=(10,5))
        plt.title("Discriminator and Generator loss during Training")
        # plot Discriminator and generator loss
        plt.plot(D_losses, label="D Loss")
        plt.plot(G_losses, label="G Loss")
        # get plot axis
        ax = plt.gca()
        # remove right and top spine
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # add labels and create legend
        plt.xlabel("num_epochs")
        plt.legend()
        plt.show()


        if self.visualize == False:
            a, b = plot_classifier(self.teacher, X.max(axis=0), X.min(axis=0))
            for i in tqdm(range(len(res_student))):
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                fig.set_size_inches(20, 5.8)
                ax1.plot(a_student[i], b_student[i], '-r', label='Optimizer Classifier')
                ax1.scatter(X[:, 0], X[:, 1], c=Y)
                ax1.scatter(generated_samples[:i+1, 0], generated_samples[:i+1, 1], c=generated_labels[:i+1], marker='x')
                ax1.legend(loc="upper right")
                ax1.set_title("Data Generation (Optimizer)")
                #ax1.set_xlim([X.min()-0.5, X.max()+0.5])
                #ax1.set_ylim([X.min()-0.5, X.max()+0.5])

                ax2.plot(a_example[i], b_example[i], '-g', label='SGD Classifier')
                ax2.scatter(X[:, 0], X[:, 1], c=Y)
                ax2.scatter(selected_samples[:i+1, 0], selected_samples[:i+1, 1], c=selected_labels[:i+1], marker='x')
                ax2.legend(loc="upper right")
                ax2.set_title("Data Selection (IMT)")
                # ax2.set_xlim([X.min()-0.5, X.max()+0.5])
                # ax2.set_xlim([X.min()-0.5, X.max()+0.5])

                ax3.plot(res_example, 'go', label="linear classifier", alpha=0.5)
                ax3.plot(res_baseline[:i+1], 'bo', label="%s & baseline" % self.opt.teaching_mode, alpha=0.5)
                ax3.plot(res_student[:i+1], 'ro', label="%s & linear classifier" % self.opt.teaching_mode, alpha=0.5)
                # ax3.axhline(y=teacher_acc, color='k', linestyle='-', label="teacher accuracy")
                ax3.legend(loc="upper right")
                ax3.set_title("W Difference")
                #ax3.set_aspect('equal')

                plt.savefig(CONF.PATH.OUTPUT + "/file%02d.png" % i)

                plt.close()

            os.chdir(CONF.PATH.OUTPUT)
            subprocess.call([
                'ffmpeg', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
                'video_name.mp4'
            ])
            for file_name in glob.glob("*.png"):
                os.remove(file_name)
