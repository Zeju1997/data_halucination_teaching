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
import os
import networks
import data
from tqdm import tqdm
import torchvision
from torchvision import transforms
from torchvision.transforms import ToTensor
import torchvision.utils as vutils
from torchvision.utils import save_image, make_grid
from train_utils import *
from eval import EvalMetrics
import teachers.omniscient_teacher as omniscient
import teachers.surrogate_teacher as surrogate
import teachers.imitation_teacher as imitation
import teachers.utils as utils
import matplotlib.pyplot as plt
import data.dataset_loader as data_loader

from datasets import BaseDataset

import networks.cgan as cgan
import networks.unrolled_optimizer as unrolled

from sklearn.datasets import make_moons, make_classification
from sklearn.model_selection import train_test_split

import subprocess
import glob

sys.path.append('..') #Hack add ROOT DIR
from baseconfig import CONF

torch.manual_seed(0)


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
        # torch.nn.init.kaiming_uniform(m.weight)
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


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.opt.log_path = os.path.join(CONF.PATH.LOG, self.opt.model_name)

        self.visualize = True

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        if self.opt.data_mode == "cifar10":
            self.train_dataset = torchvision.datasets.CIFAR10(root=CONF.PATH.DATA, train=True, download=True, transform=ToTensor())
            self.test_dataset = torchvision.datasets.CIFAR10(root=CONF.PATH.DATA, train=False, download=True, transform=ToTensor())
            # self.train_loader = DataLoader(self.train_dataset, batch_size=self.opt.batch_size)
            self.train_loader = DataLoader(self.train_dataset, batch_size=len(self.train_dataset))
            # self.test_loader = DataLoader(self.test_dataset, batch_size=self.opt.batch_size)
            self.test_loader = DataLoader(self.test_dataset, batch_size=len(self.test_dataset))
        elif self.opt.data_mode == "mnist":
            self.train_dataset = torchvision.datasets.MNIST(
                root=CONF.PATH.DATA,
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                ))
            self.test_dataset = torchvision.datasets.MNIST(
                root=CONF.PATH.DATA,
                train=False,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                ))
        elif self.opt.data_mode == "gaussian":
            print("Generating Gaussian data ...")
        elif self.opt.data_mode == "moon":
            print("Generating moon data ...")
        elif self.opt.data_mode == "linearly_seperable":
            print("Generating linearly seperable data ...")
        else:
            print("Unrecognized data!")
            sys.exit()

        self.get_teacher_student()

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.opt.log_path, mode))

    def get_teacher_student(self):
        if self.opt.teaching_mode == "omniscient":
            if self.opt.data_mode == "cifar10":
                self.teacher = omniscient.OmniscientConvTeacher(self.opt.eta)
                self.student = omniscient.OmniscientConvStudent(self.opt.eta)
            else: # mnist / gaussian / moon
                self.teacher = omniscient.OmniscientLinearTeacher(self.opt.dim)
                torch.save(self.teacher.state_dict(), 'pretrained/teacher_w0.pth')
                # self.teacher.load_state_dict(torch.load('pretrained/teacher_w0.pth'))

                self.student = omniscient.OmniscientLinearStudent(self.opt.dim)

                self.baseline = omniscient.OmniscientLinearStudent(self.opt.dim)

                # self.teacher = omniscient.TeacherClassifier(self.opt.dim)
                # self.student = omniscient.StudentClassifier(self.opt.dim)
        elif self.opt.teaching_mode == "surrogate":
            if self.opt.data_mode == "cifar10":
                if self.opt.same_feat_space:
                    self.teacher = surrogate.SurrogateConvTeacher(self.opt.eta)
                    self.student = surrogate.SurrogateConvStudent(self.opt.eta)
                else:
                    self.teacher = surrogate.SurrogateConvTeacher(self.opt.eta)
                    self.student = surrogate.SurrogateConvStudent(self.opt.eta)
            elif self.opt.data_mode == "mnist":
                if self.opt.same_feat_space:
                    self.teacher = surrogate.SurrogateLinearTeacher(self.opt.dim)
                    self.student = surrogate.SurrogateLinearStudent(self.opt.dim)
                else:
                    self.teacher = surrogate.SurrogateDiffLinearTeacher(self.opt.dim, 24, normal_dist=True)
                    self.student = surrogate.SurrogateLinearStudent(self.opt.dim)
        elif self.opt.teaching_mode == "imitation":
            if self.opt.data_mode == "cifar10":
                if self.opt.same_feat_space:
                    tmp = next(iter(self.train_loader))[0]
                    fst_x = torch.Tensor(tmp[torch.randint(0, tmp.shape[0], (1,)).item()]).unsqueeze(0).cuda()
                    self.teacher = imitation.ImmitationConvTeacher(self.opt.eta, fst_x)
                    self.student = utils.BaseConv(self.opt.eta)
                else:
                    fst_x = torch.Tensor(data[torch.randint(0, data.shape[0], (1,)).item()]).unsqueeze(0).cuda()
                    self.teacher = imitation.ImmitationConvTeacher(self.opt.eta, fst_x)
                    self.student = utils.BaseConv(self.opt.eta)
            elif self.opt.data_mode == "mnist":
                if self.opt.same_feat_space:
                    fst_x = torch.Tensor(self.opt.dim).cuda()
                    self.teacher = imitation.ImitationLinearTeacher(self.opt.dim, fst_x)
                    self.student = utils.BaseLinear(self.opt.dim)
                else:
                    fst_x = torch.Tensor(self.opt.dim).cuda()
                    self.teacher = imitation.ImitationDiffLinearTeacher(self.opt.eta, fst_x)
                    self.student = utils.BaseConv(self.opt.eta)
        else:
            print("Unrecognized teacher!")
            sys.exit()

        self.student.load_state_dict(torch.load('pretrained/teacher_w0.pth'))
        self.baseline.load_state_dict(torch.load('pretrained/teacher_w0.pth'))

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

    def train(self):
        """Run a single epoch of training and validation
        """

        print("Training")
        # self.set_train()

        if self.opt.data_mode == "cifar10":
            train_loader = DataLoader(self.train_dataset, batch_size=len(self.train_dataset), shuffle=False)
            X = next(iter(train_loader))[0].numpy()
            y = next(iter(train_loader))[1].numpy()
            (N, W, H, C) = self.train_dataset.data.shape
            dim = W*H*C
            example = utils.BaseConv(self.opt.eta)

        elif self.opt.data_mode == "mnist":
            data_loader = DataLoader(self.train_dataset, batch_size=len(self.train_dataset), shuffle=False)
            X = next(iter(data_loader))[0].numpy()
            Y = next(iter(data_loader))[1].numpy()
            (N, W, H) = self.train_dataset.data.shape
            dim = W*H
            example = utils.BaseLinear(self.opt.dim)
            tmp_student = utils.BaseLinear(self.opt.dim)
            X = X.reshape((N, dim))

            # create new data set with class 1 as 0 and class 2 as 1
            f = (Y == self.opt.class_1) | (Y == self.opt.class_2)
            X = X[f]
            Y = Y[f]
            Y = np.where(Y == self.opt.class_1, 0, 1)

        elif self.opt.data_mode == "gaussian":
            dim__diff = 7
            nb_data_per_class = 1000

            X, Y = self.init_data(self.opt.dim, nb_data_per_class)

            example = utils.BaseLinear(self.opt.dim)
            tmp_student = utils.BaseLinear(self.opt.dim)

            if self.visualize:
                fig = plt.figure(figsize=(8, 5))
                a, b = plot_classifier(self.teacher, X.max(axis=0), X.min(axis=0))
                plt.plot(a, b, '-r', label='y=wx+b')
                plt.scatter(X[:, 0], X[:, 1], c=Y)
                plt.title('Gaussian Data')
                #plt.show()
                plt.close()

        elif self.opt.data_mode == "moon":
            np.random.seed(0)
            noise_val = 0.2

            X, Y = make_moons(self.opt.nb_train+self.opt.nb_test, noise=noise_val)

            example = utils.BaseLinear(self.opt.dim)
            tmp_student = utils.BaseLinear(self.opt.dim)

            if self.visualize:
                fig = plt.figure(figsize=(8, 5))
                a, b = plot_classifier(self.teacher, X.max(axis=0), X.min(axis=0))
                plt.plot(a, b, '-r', label='y=wx+b')
                plt.scatter(X[:, 0], X[:, 1], c=Y)
                plt.title('Moon Data')
                # plt.show()
                plt.close()

        elif self.opt.data_mode == "linearly_seperable":
            X, Y = make_classification(
                n_samples=self.opt.nb_train+self.opt.nb_test, n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
            )
            rng = np.random.RandomState(2)
            X += 2 * rng.uniform(size=X.shape)

            example = utils.BaseLinear(self.opt.dim)
            tmp_student = utils.BaseLinear(self.opt.dim)

            if self.visualize:
                fig = plt.figure(figsize=(8, 5))
                a, b = plot_classifier(self.teacher, X.max(axis=0), X.min(axis=0))
                plt.plot(a, b, '-r', label='y=wx+b')
                plt.scatter(X[:, 0], X[:, 1], c=Y)
                plt.title('Linearly Seperable Data')
                # plt.show()
                plt.close()

        else:
            print("Unrecognized data mode!")
            sys.exit()

        example.load_state_dict(self.teacher.state_dict())
        tmp_student.load_state_dict(self.teacher.state_dict())

        # Shuffle datasets
        randomize = np.arange(X.shape[0])
        np.random.shuffle(randomize)
        X = X[randomize]
        Y = Y[randomize]

        nb_batch = int(self.opt.nb_train / self.opt.batch_size)

        if self.opt.data_mode == "cifar10":
            X_train = torch.tensor(X[:self.opt.nb_train])
            Y_train = torch.tensor(Y[:self.opt.nb_train], dtype=torch.long)
            X_test = torch.tensor(X[self.opt.nb_train:self.opt.nb_train + self.opt.nb_test])
            Y_test = torch.tensor(Y[self.opt.nb_train:self.opt.nb_train + self.opt.nb_test], dtype=torch.long)
        elif self.opt.data_mode == "mnist":
            X_train_im = torch.tensor(X[:self.opt.nb_train], dtype=torch.float)
            Y_train = torch.tensor(Y[:self.opt.nb_train], dtype=torch.float)
            X_test_im = torch.tensor(X[self.opt.nb_train:self.opt.nb_train + self.opt.nb_test], dtype=torch.float)
            Y_test = torch.tensor(Y[self.opt.nb_train:self.opt.nb_train + self.opt.nb_test], dtype=torch.float)

            proj_matrix = torch.empty(X.shape[1], self.opt.dim).normal_(mean=0, std=0.1)
            X_train = X_train_im @ proj_matrix
            X_test = X_test_im @ proj_matrix

            data_train_im = BaseDataset(X_train_im, Y_train)
            data_test_im = BaseDataset(X_test_im, Y_test)
            train_loader_im = DataLoader(data_train_im, batch_size=self.opt.batch_size, drop_last=True)
            test_loader_im = DataLoader(data_test_im, batch_size=self.opt.batch_size, drop_last=True)

        else:
            X_train = torch.tensor(X[:self.opt.nb_train], dtype=torch.float)
            Y_train = torch.tensor(Y[:self.opt.nb_train], dtype=torch.float)
            X_test = torch.tensor(X[self.opt.nb_train:self.opt.nb_train + self.opt.nb_test], dtype=torch.float)
            Y_test = torch.tensor(Y[self.opt.nb_train:self.opt.nb_train + self.opt.nb_test], dtype=torch.float)

        data_train = BaseDataset(X_train, Y_train)
        data_test = BaseDataset(X_test, Y_test)
        train_loader = DataLoader(data_train, batch_size=self.opt.batch_size, drop_last=True)
        test_loader = DataLoader(data_test, batch_size=self.opt.batch_size, drop_last=True)

        '''
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Plot some training images
        real_batch = next(iter(train_loader))

        if self.visualize == True:
            plt.figure(figsize=(8, 8))
            plt.axis("off")
            plt.title("Training Images")
            plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
            plt.show()

        adversarial_loss = torch.nn.MSELoss()

        # Create the generator
        netG = cgan.Generator(self.opt).cuda()
        netG.apply(weights_init)

        netD = cgan.Discriminator(self.opt).cuda()
        netD.apply(weights_init)

        optimizer_D = optim.Adam(netD.parameters(), lr=self.opt.lr, betas=(self.opt.b1, self.opt.b2))
        optimizer_G = optim.Adam(netG.parameters(), lr=self.opt.lr, betas=(self.opt.b1, self.opt.b2))
        self.step = 0
        for epoch in range(self.opt.n_epochs):
            for i, (imgs, labels) in enumerate(train_loader):

                # Adversarial ground truths
                valid = Variable(torch.cuda.FloatTensor(self.opt.batch_size, 1).fill_(1.0), requires_grad=False)
                fake = Variable(torch.cuda.FloatTensor(self.opt.batch_size, 1).fill_(0.0), requires_grad=False)

                # Configure input
                real_imgs = Variable(imgs.type(torch.cuda.FloatTensor))
                labels = Variable(labels.type(torch.cuda.LongTensor))

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Sample noise and labels as generator input
                z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (self.opt.batch_size, self.opt.dim))))

                # self.opt.n_classes = 2
                indices = np.random.randint(2, size=self.opt.batch_size).astype(int)
                # classes = np.array([self.opt.class_1, self.opt.class_2])
                classes = np.array([0, 1])
                labels = classes[indices]
                gen_labels = Variable(torch.cuda.LongTensor(labels))
                labels = Variable(torch.cuda.LongTensor(labels))
                # gen_labels = Variable(torch.cuda.LongTensor(np.random.randint(0, self.opt.n_classes, batch_size)))

                # z = torch.cat((z, gen_labels.unsqueeze(0)), dim=1)
                # Generate a batch of images
                gen_imgs = netG(z, gen_labels)

                # Loss measures generator's ability to fool the discriminator
                validity = netD(gen_imgs, gen_labels)
                g_loss = adversarial_loss(validity, valid)

                g_loss.backward()
                optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Loss for real images
                validity_real = netD(real_imgs, labels)
                d_real_loss = adversarial_loss(validity_real, valid)

                # Loss for fake images
                validity_fake = netD(gen_imgs.detach(), gen_labels)
                d_fake_loss = adversarial_loss(validity_fake, fake)

                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2

                d_loss.backward()
                optimizer_D.step()

                self.step = self.step + 1

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, self.opt.n_epochs, i, len(train_loader), d_loss.item(), g_loss.item())
                )

            if epoch % 2 == 0:
                n_row = 10
                # Sample noise
                # z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (n_row ** 2, self.opt.dim))))
                z = Variable(torch.randn(n_row ** 2, self.opt.dim)).cuda()

                indices = np.random.randint(2, size=100).astype(int)
                classes = np.array([0, 1])
                labels = classes[indices]
                labels = Variable(torch.cuda.LongTensor(labels))

                gen_imgs = netG(z, labels)

                # unproj_matrix = np.linalg.pinv(proj_matrix)
                # im = gen_imgs.detach().clone().cpu() @ unproj_matrix

                im = gen_imgs.detach().clone().cpu()

                # img_shape = (1, 28, 28)
                # im = np.reshape(im, (im.shape[0], 28, 28))
                # im = im.view(im.size(0), *img_shape)
                # im = torch.from_numpy(im)
                # im = im.unsqueeze(1)
                # im = np.transpose(im, (1, 2, 0))

                save_folder = os.path.join(self.opt.log_path, "imgs")
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)

                grid = make_grid(im, nrow=10, normalize=True)
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(grid.permute(1, 2, 0).data, cmap='binary')
                ax.axis('off')
                plt.title("Fake Images, Label", )
                img_path = os.path.join(save_folder, "results_{}_imgs.png".format(epoch))
                plt.savefig(img_path)
                plt.close()
                
        sys.exit()
        '''

        # train teacher
        accuracies = []
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.teacher.optim, milestones=[80, 160], gamma=0.1)
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
            self.scheduler.step()

        if self.visualize == True:
            fig = plt.figure()
            plt.plot(accuracies, c="b", label="Teacher (CNN)")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.close()

            fig = plt.figure(figsize=(8, 5))
            a, b = plot_classifier(self.teacher, X.max(axis=0), X.min(axis=0))
            plt.plot(a, b, '-r', label='y=wx+b')
            plt.scatter(X[:, 0], X[:, 1], c=Y)
            plt.title('Initial Classifer Weight')
            plt.close()

        w_star = self.teacher.lin.weight

        # train example
        res_example = []
        a_example= []
        b_example = []
        w_diff_example = []
        for idx in tqdm(range(self.opt.n_iter)):
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

                example.update(data, label)

            example.eval()
            test = example(X_test.cuda()).cpu()

            a, b = plot_classifier(example, X.max(axis=0), X.min(axis=0))
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
            res_example.append(acc)

            diff = torch.linalg.norm(w_star - example.lin.weight, ord=2) ** 2
            w_diff_example.append(diff.detach().clone().cpu())

            if acc > 0.6 and idx == 0:
                sys.exit()

        # plt.plot(res_example, c="b", label="Example (CNN)")
        plt.plot(w_diff_example, c='g', label="weight diff")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        # plt.show()
        plt.close()

        # train baseline
        res_baseline = []
        a_baseline = []
        b_baseline = []
        w_diff_baseline = []
        for t in tqdm(range(self.opt.n_iter)):
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
            acc_base = nb_correct / X_test.size(0)
            res_baseline.append(acc_base)

            diff = torch.linalg.norm(w_star - self.baseline.lin.weight, ord=2) ** 2
            w_diff_baseline.append(diff.detach().clone().cpu())

            sys.stdout.write("\r" + str(t) + "/" + str(self.opt.n_iter) + ", idx=" + str(i) + " " * 100)
            sys.stdout.flush()

        print("Base line trained\n")

        # train student
        netG = unrolled.Generator(self.opt, self.teacher, tmp_student).cuda()
        netG.apply(weights_init)

        netD = unrolled.Discriminator(self.opt).cuda()
        netD.apply(weights_init)

        optimizer_D = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999), eps=1e-08, weight_decay=1e-04, amsgrad=False)
        optimizer_G = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999), eps=1e-08, weight_decay=1e-04, amsgrad=False)

        # adversarial_loss = torch.nn.BCELoss()
        adversarial_loss = torch.nn.MSELoss()

        unrolled_optimizer = unrolled.UnrolledOptimizer(opt=self.opt, teacher=self.teacher, student=tmp_student, generator=netG, X=X_train.cuda(), y=Y_train.cuda(), proj_matrix=proj_matrix)

        self.step = 0

        loss_student = []

        img_shape = (1, 28, 28)

        w_init = self.student.lin.weight
        for epoch in tqdm(range(self.opt.n_epochs)):
            if epoch != 0:
                for i, (data, labels) in enumerate(train_loader_im):
                    self.step = self.step + 1
                    # Adversarial ground truths
                    valid = Variable(torch.cuda.FloatTensor(self.opt.batch_size, 1).fill_(1.0), requires_grad=False)
                    fake = Variable(torch.cuda.FloatTensor(self.opt.batch_size, 1).fill_(0.0), requires_grad=False)

                    # Configure input
                    real_samples = Variable(data.type(torch.cuda.FloatTensor))
                    # real_samples = data.view(data.size(0), *img_shape)
                    # real_samples = Variable(real_samples.type(torch.cuda.FloatTensor))
                    real_labels = Variable(labels.type(torch.cuda.LongTensor))

                    # -----------------
                    #  Train Generator
                    # -----------------

                    optimizer_G.zero_grad()

                    # Generate a batch of images
                    # w_stu = self.student.lin.weight

                    # i = torch.randint(0, nb_batch, size=(1,)).item()
                    # i_min = i * self.opt.batch_size
                    # i_max = (i + 1) * self.opt.batch_size

                    # gt_x = X_train[i_min:i_max].cuda()
                    # generated_labels = Y_train[i_min:i_max].cuda()

                    # x = torch.cat((w_stu, w_stu-w_star, gt_x, generated_labels.unsqueeze(0)), dim=1)
                    # generated_samples = netG(x)

                    # Loss measures generator's ability to fool the discriminator
                    # validity = netD(generated_samples, Variable(generated_labels.type(torch.cuda.LongTensor)))
                    # g_loss = adversarial_loss(validity, valid)

                    # Loss measures generator's ability to fool the discriminator
                    w_t = netG.state_dict()
                    gradients, generator_loss, generated_samples, generated_labels, g_loss = unrolled_optimizer(w_t, w_star, w_init, netD, valid)

                    loss_student.append(generator_loss.item())

                    with torch.no_grad():
                        for p, g in zip(netG.parameters(), gradients):
                            p.grad = g

                    optimizer_G.step()

                    # ---------------------
                    #  Train Discriminator
                    # ---------------------

                    # for _ in range(self.opt.n_critic):
                    if i % self.opt.n_critic == 0:
                        optimizer_D.zero_grad()

                        # Loss for real images
                        validity_real = netD(real_samples, real_labels)
                        d_real_loss = adversarial_loss(validity_real, valid)

                        # Loss for fake images
                        validity_fake = netD(generated_samples.detach(), Variable(generated_labels.type(torch.cuda.LongTensor)))
                        d_fake_loss = adversarial_loss(validity_fake, fake)

                        # Total discriminator loss
                        d_loss = (d_real_loss + d_fake_loss) / 2

                        d_loss.backward()
                        optimizer_D.step()

                    if i % self.opt.log_frequency == 0:
                        print(
                            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                            % (epoch, self.opt.n_epochs, i, len(train_loader), d_loss.item(), g_loss.item())
                        )
                        self.log("train", d_loss.item(), g_loss.item())

            if epoch % self.opt.save_frequency == 0 and epoch >= self.opt.start_saving:
                res_student = []
                a_student = []
                b_student = []
                w_diff_student = []

                self.student.load_state_dict(torch.load('pretrained/teacher_w0.pth'))

                generated_samples = np.zeros(2)
                for idx in tqdm(range(self.opt.n_iter)):
                    if idx != 0:
                        w_t = self.student.lin.weight

                        i = torch.randint(0, nb_batch, size=(1,)).item()
                        i_min = i * self.opt.batch_size
                        i_max = (i + 1) * self.opt.batch_size

                        gt_x = X_train[i_min:i_max].cuda()
                        y = Y_train[i_min:i_max].cuda()

                        # z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, gt_x.shape)))
                        z = Variable(torch.randn((self.opt.batch_size, self.opt.latent_dim))).cuda()

                        # x = torch.cat((w_t, w_t-w_star, gt_x, y.unsqueeze(0)), dim=1)
                        x = torch.cat((w_t, w_t-w_star, z), dim=1)
                        generated_sample = netG(x, y)

                        if idx == 1:
                            generated_samples = generated_sample.cpu().detach().numpy()  # [np.newaxis, :]
                            generated_labels = y.cpu().detach().numpy()  # [np.newaxis, :]
                        else:
                            generated_samples = np.concatenate((generated_samples, generated_sample.cpu().detach().numpy()), axis=0)
                            generated_labels = np.concatenate((generated_labels, y.cpu().detach().numpy()), axis=0)

                        generated_sample = generated_sample @ proj_matrix.cuda()
                        self.student.update(generated_sample, y)

                    self.student.eval()
                    test = self.student(X_test.cuda()).cpu()

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

                if self.opt.data_mode == "gaussian" or self.opt.data_mode == "moon":
                    self.make_results_img_2d(X, Y, a_student, b_student, generated_samples, generated_labels, w_diff_example, w_diff_baseline, w_diff_student, epoch)
                else:
                    self.make_results_img(X, Y, a_student, b_student, generated_samples, generated_labels, w_diff_example, w_diff_baseline, w_diff_student, epoch, proj_matrix)

                save_folder = os.path.join(self.opt.log_path, "models", "weights_{}".format(epoch))
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)

                save_path = os.path.join(save_folder, "netG_{}.pth".format("models", epoch))
                to_save = netG.state_dict()
                torch.save(to_save, save_path)

                save_path = os.path.join(save_folder, "netD_{}.pth".format("models", epoch))
                to_save = netD.state_dict()
                torch.save(to_save, save_path)

        plt.plot(w_diff_example, c='g', label="weight diff")
        plt.plot(w_diff_student, c='r', label="weight diff student")
        plt.plot(w_diff_baseline, c='b', label="weight diff IMT")
        # plt.plot(res_example, c='b', label="linear classifier")
        # plt.plot(res_student, c='r', label="%s & linear classifier" % self.opt.teaching_mode)
        plt.title(str(self.opt.data_mode) + "Model (class : " + str(self.opt.class_1) + ", " + str(self.opt.class_2) + ")")
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.close()

        if self.visualize == False:
            self.make_results_video(X, Y, a_student, b_student, generated_samples, generated_labels, w_diff_example, w_diff_baseline, w_diff_student)

        sys.exit()

        # Root directory for dataset
        batch_size = 64
        img_size = 28
        channels = 1
        sample_interval = 400
        workers = 8

        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        dataset = torchvision.datasets.MNIST(root=CONF.PATH.DATA, train=True, download=True, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

        # Decide which device we want to run on
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Plot some training images
        real_batch = next(iter(dataloader))

        if self.visualize == True:
            plt.figure(figsize=(8,8))
            plt.axis("off")
            plt.title("Training Images")
            plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
            plt.show()

        adversarial_loss = torch.nn.MSELoss()

        # Create the generator
        netG = cgan.Generator(self.opt).cuda()
        netD = cgan.Discriminator(self.opt).cuda()

        optimizer_D = optim.Adam(netD.parameters(), lr=self.opt.lr, betas=(self.opt.b1, self.opt.b2))
        optimizer_G = optim.Adam(netG.parameters(), lr=self.opt.lr, betas=(self.opt.b1, self.opt.b2))

        for epoch in range(self.opt.n_epochs):
            for i, (imgs, labels) in enumerate(dataloader):

                batch_size = imgs.shape[0]

                # Adversarial ground truths
                valid = Variable(torch.cuda.FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
                fake = Variable(torch.cuda.FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

                # Configure input
                real_imgs = Variable(imgs.type(torch.cuda.FloatTensor))
                labels = Variable(labels.type(torch.cuda.LongTensor))

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Sample noise and labels as generator input
                z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (batch_size, self.opt.latent_dim))))

                self.opt.n_classes = 2
                indices = np.random.randint(2, size=batch_size).astype(int)
                classes = np.array([self.opt.class_1, self.opt.class_2])
                labels = classes[indices]
                gen_labels = Variable(torch.cuda.LongTensor(labels))
                labels = Variable(torch.cuda.LongTensor(labels))
                # gen_labels = Variable(torch.cuda.LongTensor(np.random.randint(0, self.opt.n_classes, batch_size)))

                # Generate a batch of images
                gen_imgs = netG(z, gen_labels)

                # Loss measures generator's ability to fool the discriminator
                validity = netD(gen_imgs, gen_labels)
                g_loss = adversarial_loss(validity, valid)

                g_loss.backward()
                optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Loss for real images
                validity_real = netD(real_imgs, labels)
                d_real_loss = adversarial_loss(validity_real, valid)

                # Loss for fake images
                validity_fake = netD(gen_imgs.detach(), gen_labels)
                d_fake_loss = adversarial_loss(validity_fake, fake)

                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2

                d_loss.backward()
                optimizer_D.step()

                self.step = self.step + 1

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, self.opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )

                batches_done = epoch * len(dataloader) + i
                if batches_done % sample_interval == 0:
                    self.sample_image(netG, n_row=10, batches_done=batches_done)

        visualize = True
        if visualize:
            n_row = 5
            # Sample noise
            z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (n_row ** 2, self.opt.latent_dim))))
            # Get labels ranging from 0 to n_classes for n rows
            # labels = np.array([num for _ in range(n_row) for num in range(n_row)])
            indices = np.random.randint(2, size=25).astype(int)
            # classes = np.array([self.opt.class_1, self.opt.class_2])
            classes = np.array([0, 1])
            labels = classes[indices]
            labels = Variable(torch.cuda.LongTensor(labels))
            gen_imgs = netG(z, labels)
            im = np.transpose(gen_imgs[1, :, :, :].detach().cpu().numpy(), (1, 2, 0))

            # Plot the fake images from the last epoch
            plt.figure(figsize=(8,8))
            plt.axis("off")
            plt.title("Fake Images, Label", )
            plt.imshow(im[:, :, 0], cmap="gray")
            plt.show()

    def make_results_img(self, X, Y, a_student, b_student, generated_samples, generated_labels, w_diff_example, w_diff_baseline, w_diff_student, epoch, proj_matrix):
        # unproj_matrix = np.linalg.pinv(proj_matrix)
        n_rows = 10
        indices = torch.randint(0, len(generated_samples), (n_rows**2,))
        labels = generated_labels[indices]
        samples = generated_samples[indices]

        # gen_imgs = samples @ unproj_matrix

        img_shape = (1, 28, 28)
        # gen_imgs = samples
        im = np.reshape(samples, (samples.shape[0], *img_shape))
        im = torch.from_numpy(im)

        save_folder = os.path.join(self.opt.log_path, "imgs")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        grid = make_grid(im, nrow=10, normalize=True)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(grid.permute(1, 2, 0).data, cmap='binary')
        ax.axis('off')
        plt.title("Fake Images, Label", )
        img_path = os.path.join(save_folder, "results_{}_imgs.png".format(epoch))
        plt.savefig(img_path)
        plt.close()

        # plt.figure(figsize=(10, 10)) # specifying the overall grid size

        # for i in range(25):
        #    plt.subplot(5, 5, i+1)    # the number of images in the grid is 5*5 (25)
        #    plt.imshow(im[:, :, i], cmap="gray")

        # plt.axis("off")

        plt.figure(figsize=(10, 10))
        # plt.plot(res_example, 'go', label="linear classifier", alpha=0.5)
        # plt.plot(res_baseline[:i+1], 'bo', label="%s & baseline" % self.opt.teaching_mode, alpha=0.5)
        # plt.plot(res_student[:i+1], 'ro', label="%s & linear classifier" % self.opt.teaching_mode, alpha=0.5)
        plt.plot(w_diff_example, 'go', label="linear classifier", alpha=0.5)
        plt.plot(w_diff_baseline, 'bo', label="%s & baseline" % self.opt.teaching_mode, alpha=0.5)
        plt.plot(w_diff_student, 'ro', label="%s & linear classifier" % self.opt.teaching_mode, alpha=0.5)
        # plt.axhline(y=teacher_acc, color='k', linestyle='-', label="teacher accuracy")
        plt.legend(loc="upper right")
        plt.title("Test Set Accuracy")
        #plt.set_aspect('equal')

        img_path = os.path.join(save_folder, "results_{}_w_diff.png".format(epoch))
        plt.savefig(img_path)
        plt.close()

    def make_results_img_2d(self, X, Y, a_student, b_student, generated_samples, generated_labels, w_diff_example, w_diff_baseline, w_diff_student, epoch):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(14, 6)
        ax1.plot(a_student[-1], b_student[-1], '-r', label='Optimizer Classifier')
        ax1.scatter(X[:, 0], X[:, 1], c=Y)
        ax1.scatter(generated_samples[:, 0], generated_samples[:, 1], c=generated_labels[:], marker='x')
        ax1.legend(loc="upper right")
        ax1.set_title("Data Generation (Optimizer)")
        #ax1.set_xlim([X.min()-0.5, X.max()+0.5])
        #ax1.set_ylim([X.min()-0.5, X.max()+0.5])

        # ax2.plot(res_example, 'go', label="linear classifier", alpha=0.5)
        # ax2.plot(res_baseline[:i+1], 'bo', label="%s & baseline" % self.opt.teaching_mode, alpha=0.5)
        # ax2.plot(res_student[:i+1], 'ro', label="%s & linear classifier" % self.opt.teaching_mode, alpha=0.5)
        ax2.plot(w_diff_example, 'go', label="linear classifier", alpha=0.5)
        ax2.plot(w_diff_baseline, 'bo', label="%s & baseline" % self.opt.teaching_mode, alpha=0.5)
        ax2.plot(w_diff_student, 'ro', label="%s & linear classifier" % self.opt.teaching_mode, alpha=0.5)
        # ax2.axhline(y=teacher_acc, color='k', linestyle='-', label="teacher accuracy")
        ax2.legend(loc="upper right")
        ax2.set_title("Test Set Accuracy")
        #ax2.set_aspect('equal')

        save_folder = os.path.join(self.opt.log_path, "imgs")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        img_path = os.path.join(save_folder, "results_{}.png".format(epoch))
        plt.savefig(img_path)
        plt.close()


    def make_results_video(self, X, Y, a_student, b_student, generated_samples, generated_labels, w_diff_example, w_diff_baseline, w_diff_student):
        # a, b = plot_classifier(self.teacher, X.max(axis=0), X.min(axis=0))
        for i in tqdm(range(len(w_diff_student))):
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(14, 6)
            ax1.plot(a_student[i], b_student[i], '-r', label='Optimizer Classifier')
            ax1.scatter(X[:, 0], X[:, 1], c=Y)
            ax1.scatter(generated_samples[:i+1, 0], generated_samples[:i+1, 1], c=generated_labels[:i+1], marker='x')
            ax1.legend(loc="upper right")
            ax1.set_title("Data Generation (Optimizer)")
            #ax1.set_xlim([X.min()-0.5, X.max()+0.5])
            #ax1.set_ylim([X.min()-0.5, X.max()+0.5])

            # ax2.plot(a_example[i], b_example[i], '-g', label='SGD Classifier')
            # ax2.scatter(X[:, 0], X[:, 1], c=Y)
            # ax2.scatter(selected_samples[:i+1, 0], selected_samples[:i+1, 1], c=selected_labels[:i+1], marker='x')
            # ax2.legend(loc="upper right")
            # ax2.set_title("Data Selection (IMT)")
            # ax2.set_xlim([X.min()-0.5, X.max()+0.5])
            # ax2.set_xlim([X.min()-0.5, X.max()+0.5])

            # ax2.plot(res_example, 'go', label="linear classifier", alpha=0.5)
            # ax2.plot(res_baseline[:i+1], 'bo', label="%s & baseline" % self.opt.teaching_mode, alpha=0.5)
            # ax2.plot(res_student[:i+1], 'ro', label="%s & linear classifier" % self.opt.teaching_mode, alpha=0.5)
            ax2.plot(w_diff_example, 'go', label="linear classifier", alpha=0.5)
            ax2.plot(w_diff_baseline[:i+1], 'bo', label="%s & baseline" % self.opt.teaching_mode, alpha=0.5)
            ax2.plot(w_diff_student[:i+1], 'ro', label="%s & linear classifier" % self.opt.teaching_mode, alpha=0.5)
            # ax2.axhline(y=teacher_acc, color='k', linestyle='-', label="teacher accuracy")
            ax2.legend(loc="upper right")
            ax2.set_title("Test Set Accuracy")
            #ax2.set_aspect('equal')

            plt.savefig(CONF.PATH.OUTPUT + "/file%02d.png" % i)

            plt.close()

        os.chdir(CONF.PATH.OUTPUT)
        subprocess.call([
            'ffmpeg', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
            'video_name.mp4'
        ])
        for file_name in glob.glob("*.png"):
            os.remove(file_name)

    def log(self, mode, d_loss, g_loss):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        writer.add_scalar("d_loss/{}".format("sa"), d_loss, self.step)
        writer.add_scalar("g_loss/{}".format("as"), g_loss, self.step)

    def main(self):
        X_test = next(iter(self.test_loader))[0].numpy()
        Y_test = next(iter(self.test_loader))[1].numpy()

        accuracies = []
        for epoch in tqdm(range(100)):
            print('\nEpoch: %d' % epoch)
            self.teacher.train()
            train_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.teacher.update(inputs, targets)

                outputs = self.teacher(inputs.cuda())
                predicted = torch.max(outputs, dim=1).indices

                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # print('Acc: %.3f%% (%d/%d)'% (100.*correct/total, correct, total))

            self.teacher.eval()
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    outputs = self.teacher(inputs.cuda())
                    predicted = torch.max(outputs, dim=1).indices

                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            # Save checkpoint.
            acc = 100.*correct/total
            accuracies.append(acc)

            print("Epoch", epoch, "Acc", acc)
        plt.plot(accuracies, c="b", label="Teacher (CNN)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()
        '''
            test = self.teacher(X_test.cuda()).cpu()
            tmp = torch.where(test > 0.5, torch.ones(1), torch.zeros(1))
            nb_correct = torch.where(tmp.view(-1) == Y_test, torch.ones(1), torch.zeros(1)).sum().item()
            accuracies.append(nb_correct / X_test.size(0))

        plt.plot(accuracies, c="b", label="Teacher (CNN)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()
        '''

    def process_batch(self, inputs):
        #for key, ipt in inputs.items():
        #    if key != 'case_name':
        #        inputs[key] = ipt.to(self.device)

        outputs = {}

        # features = self.models["encoder"](inputs["image"])
        # preds = self.models["decoder"](features)
        preds = self.models["unet"](inputs["image"].to(self.device))

        outputs["pred"] = preds
        outputs["pred_idx"] = torch.argmax(preds, dim=1, keepdim=True)

        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)
            self.compute_accuracy(inputs, outputs, losses)
            self.log("val", inputs, outputs, losses)

            del inputs, outputs, losses

        self.set_train()

    def compute_losses(self, inputs, outputs):
        losses = {}
        total_loss = 0

        pred = outputs['pred']
        target = inputs['label']

        #to_optimise = self.criterion(output=pred,
        #                             target=target)

        to_optimise = self.criterion(pred, target.type(torch.LongTensor).cuda())

        total_loss += to_optimise
        losses["loss"] = total_loss
        return losses

    def compute_accuracy(self, inputs, outputs, losses):
        with torch.no_grad():
            # acc_dice = 0
            # acc_iou = 0
            for cls in range(1, self.opt.n_classes):
                fluid = self.cls_to_fluid[cls]
                losses["accuracy/dice_{}".format(fluid)] = self.eval_metric.compute_coef(outputs["pred_idx"].cpu().data,
                                                                                         inputs['label'].cpu().data,
                                                                                         mode='dice',
                                                                                         cls=cls)
                # acc_dice += losses["accuracy/dice_{}".format(fluid)]

                losses["accuracy/iou_{}".format(fluid)] = self.eval_metric.compute_coef(outputs["pred_idx"].cpu().data,
                                                                                        inputs['label'].cpu().data,
                                                                                        mode='iou',
                                                                                        cls=cls)
                # acc_iou += losses["accuracy/iou_{}".format(fluid)]


    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.opt.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.opt.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            torch.save(to_save, save_path)

        if self.epoch >= self.opt.start_saving_optimizer:
            save_path = os.path.join(save_folder, "{}.pth".format("adam"))
            torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
                                     self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                       " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))
