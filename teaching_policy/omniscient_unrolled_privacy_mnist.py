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
from tqdm import tqdm
import torchvision
from torchvision import transforms
from torchvision.transforms import ToTensor
import torchvision.utils as vutils
from torchvision.utils import save_image
from train_utils import *
import teachers.omniscient_teacher as omniscient
import teachers.utils as utils
import matplotlib.pyplot as plt

import networks.cgan as cgan
import networks.unrolled_privacy as unrolled

from datasets import BaseDataset

import csv

from utils.data import init_data, load_experiment_result, plot_graphs, initialize_weights
from utils.visualize import make_results_video, make_results_video_2d, make_results_img, make_results_img_2d

from experiments import SGDTrainer, IMTTrainer, WSTARTrainer

sys.path.append('..') #Hack add ROOT DIR
from baseconfig import CONF

from networks.resnet import ResNet50


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


class Trainer:
    def __init__(self, options):
        self.opt = options

        self.opt.model_name = "omniscient_privacy_" + self.opt.data_mode

        self.opt.log_path = os.path.join(CONF.PATH.LOG, self.opt.model_name)
        if not os.path.exists(self.opt.log_path):
            os.makedirs(self.opt.log_path)

        self.visualize = True

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.get_teacher_student()

    def get_teacher_student(self):
        if self.opt.data_mode == "cifar10":
            self.teacher = omniscient.OmniscientConvTeacher(self.opt.eta)
            self.student = omniscient.OmniscientConvStudent(self.opt.eta)
        else: # mnist / gaussian / moon
            self.teacher = omniscient.OmniscientLinearTeacher(self.opt.dim)
            self.teacher.apply(initialize_weights)
            self.student = omniscient.OmniscientLinearStudent(self.opt.dim)
            self.baseline = omniscient.OmniscientLinearStudent(self.opt.dim)
            torch.save(self.teacher.state_dict(), 'teacher_w0.pth')

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

    def main(self):
        """Run a single epoch of training and validation
        """

        print("Training")
        # self.set_train()

        if self.opt.init_data:
            init_data(self.opt)

        X = torch.load('X.pt')
        Y = torch.load('Y.pt')

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
            X_train = torch.tensor(X[:self.opt.nb_train], dtype=torch.float)
            Y_train = torch.tensor(Y[:self.opt.nb_train], dtype=torch.float)
            X_test = torch.tensor(X[self.opt.nb_train:self.opt.nb_train + self.opt.nb_test], dtype=torch.float)
            Y_test = torch.tensor(Y[self.opt.nb_train:self.opt.nb_train + self.opt.nb_test], dtype=torch.float)
            X_val = torch.tensor(X[self.opt.nb_train+self.opt.nb_test:self.opt.nb_train+self.opt.nb_test+self.opt.nb_val], dtype=torch.float)
            Y_val = torch.tensor(Y[self.opt.nb_train+self.opt.nb_test:self.opt.nb_train+self.opt.nb_test+self.opt.nb_val], dtype=torch.float)

            data_train = BaseDataset(X_train, Y_train)
            train_loader = DataLoader(data_train, batch_size=self.opt.batch_size, drop_last=True, shuffle=True)

            data_val = BaseDataset(X_val, Y_val)
            val_loader = DataLoader(data_val, batch_size=1, drop_last=True, shuffle=True)

            X_train = X_train.reshape((self.opt.nb_train, self.opt.img_size**2))
            X_test = X_test.reshape((self.opt.nb_test, self.opt.img_size**2))

            proj_matrix = torch.empty(self.opt.img_size**2, self.opt.dim).normal_(mean=0, std=0.1)
            X_train = X_train @ proj_matrix
            X_test = X_test @ proj_matrix

        else:
            X_train = torch.tensor(X[:self.opt.nb_train], dtype=torch.float)
            Y_train = torch.tensor(Y[:self.opt.nb_train], dtype=torch.float)
            X_test = torch.tensor(X[self.opt.nb_train:self.opt.nb_train + self.opt.nb_test], dtype=torch.float)
            Y_test = torch.tensor(Y[self.opt.nb_train:self.opt.nb_train + self.opt.nb_test], dtype=torch.float)

        net = ResNet50().eval().cuda()
        feat_privacy_set = torch.empty(len(val_loader), 2048).cuda()
        for idx, (images, labels) in tqdm(enumerate(val_loader)):
            images, labels = images.cuda(), labels.cuda()
            feat_privacy_set[idx, :] = net(images)

        # ---------------------
        #  Train Teacher
        # ---------------------

        if self.opt.train_wstar == True:
            wstar_trainer = WSTARTrainer(self.opt, X_train, Y_train, X_test, Y_test)
            wstar_trainer.train(self.teacher)

        self.teacher.load_state_dict(torch.load('teacher_wstar.pth'))
        w_star = self.teacher.lin.weight
        w_star = w_star / torch.norm(w_star)


        # ---------------------
        #  Train SGD
        # ---------------------

        self.opt.experiment = "SGD"
        if self.opt.train_sgd == True:

            sgd_example = utils.BaseLinear(self.opt.dim)
            sgd_example.load_state_dict(torch.load('teacher_w0.pth'))

            sgd_trainer = SGDTrainer(self.opt, X_train, Y_train, X_test, Y_test, data_train)
            random_samples, random_labels = sgd_trainer.train(sgd_example, w_star)

        res_sgd, w_diff_sgd = load_experiment_result(self.opt)

        # ---------------------
        #  Train IMT Baseline
        # ---------------------

        self.opt.experiment = "IMT_Baseline"
        if self.opt.train_baseline == True:
            self.baseline.load_state_dict(torch.load('teacher_w0.pth'))

            imt_trainer = IMTTrainer(self.opt, X_train, Y_train, X_test, Y_test, data_train)
            selected_samples, selected_labels = imt_trainer.train(self.baseline, self.teacher, w_star)

        res_baseline, w_diff_baseline = load_experiment_result(self.opt)

        # ---------------------
        #  Train Student
        # ---------------------

        self.opt.experiment = "Student"
        print("Start training {} ...".format(self.opt.experiment))
        logname = os.path.join(self.opt.log_path, 'results' + '_' + self.opt.experiment + '_' + str(self.opt.seed) + '.csv')
        if not os.path.exists(logname):
            with open(logname, 'w') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow(['iter', 'test acc', 'w diff'])

        tmp_student = utils.BaseLinear(self.opt.dim)

        if self.opt.data_mode == "mnist":
            netG = unrolled.Generator(self.opt, self.teacher, tmp_student).cuda()
            unrolled_optimizer = unrolled.UnrolledOptimizer(opt=self.opt, teacher=self.teacher, student=tmp_student, generator=netG, feature_extractor=net, X=X_train.cuda(), Y=Y_train.cuda(), X_val=X_val.cuda(), Y_val=Y_val.cuda(), feat_privacy_set=feat_privacy_set.cuda(), proj_matrix=proj_matrix)
        else:
            netG = unrolled.Generator_moon(self.opt, self.teacher, tmp_student).cuda()
            unrolled_optimizer = unrolled.UnrolledOptimizer_moon(opt=self.opt, teacher=self.teacher, student=tmp_student, generator=netG, X=X_train.cuda(), Y=Y_train.cuda())

        netG.train()
        netG.apply(weights_init)
        optimG = torch.optim.Adam(netG.parameters(), lr=self.opt.netG_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-04, amsgrad=False)

        res_student = []
        a_student = []
        b_student = []
        loss_student = []
        w_diff_student = []

        # tmp_student.load_state_dict(torch.load('teacher_w0.pth'))
        # w_init = tmp_student.state_dict()
        for _ in tqdm(range(self.opt.n_unroll)):

            w_t = netG.state_dict()
            gradients, loss = unrolled_optimizer(w_t, w_star)

            loss_student.append(loss.item())
            # loss_student = loss_student + train_loss

            with torch.no_grad():
                for p, g in zip(netG.parameters(), gradients):
                    p.grad = g

            optimG.step()

        cls = torch.arange(self.opt.n_classes)
        onehot = torch.zeros(self.opt.n_classes, self.opt.n_classes).scatter_(1, cls.view(self.opt.n_classes, 1), 1)

        self.student.load_state_dict(torch.load('teacher_w0.pth'))
        for idx in tqdm(range(self.opt.n_iter)):
            if idx != 0:
                w_t = self.student.lin.weight
                w_t = w_t / torch.norm(w_t)

                i = torch.randint(0, nb_batch, size=(1,)).item()
                gt_x, gt_y = self.data_sampler(X_train, Y_train, i)
                gt_y_onehot = onehot[gt_y.long()].cuda()

                z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, gt_x.shape)))

                # gt_x = gt_x / torch.norm(gt_x)
                x = torch.cat((w_t, w_t-w_star, gt_x), dim=1)
                # x = torch.cat((w_t, w_t-w_star), dim=1)
                generated_sample = netG(x, gt_y_onehot)

                if idx == 1:
                    generated_samples = generated_sample.cpu().detach().numpy()  # [np.newaxis, :]
                    generated_labels = gt_y.unsqueeze(1).cpu().detach().numpy()  # [np.newaxis, :]
                else:
                    generated_samples = np.concatenate((generated_samples, generated_sample.cpu().detach().numpy()), axis=0)
                    generated_labels = np.concatenate((generated_labels, gt_y.unsqueeze(1).cpu().detach().numpy()), axis=0)

                if self.opt.data_mode == "mnist":
                    generated_sample = generated_sample.reshape((self.opt.batch_size, self.opt.img_size**2))
                    generated_sample = generated_sample.detach().clone() @ proj_matrix.cuda()

                self.student.update(generated_sample.detach(), gt_y.unsqueeze(1))


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

            w = self.student.lin.weight
            w = w / torch.norm(w)
            diff = torch.linalg.norm(w_star - w, ord=2) ** 2
            w_diff_student.append(diff.detach().clone().cpu())

            with open(logname, 'a') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow([idx, acc, diff.item()])

            print("acc", acc)

        self.perceptual_loss(random_samples, selected_samples, generated_samples, feat_privacy_set, net, proj_matrix)

        if self.opt.data_mode == "gaussian" or self.opt.data_mode == "moon":
            make_results_img_2d(self.opt, X, Y, generated_samples, generated_labels, res_sgd, res_baseline, res_student, w_diff_sgd, w_diff_baseline, w_diff_student, 0, self.opt.seed)
            # make_results_video_2d(self.opt, X, Y, generated_samples, generated_labels, res_sgd, res_baseline, res_student, w_diff_sgd, w_diff_baseline, w_diff_student, 0, self.opt.seed)
        else:
            make_results_img(self.opt, X, Y, generated_samples, generated_labels, res_sgd, res_baseline, res_student, w_diff_sgd, w_diff_baseline, w_diff_student, 0, self.opt.seed)
            # make_results_video(self.opt, X, Y, generated_samples, generated_labels, res_sgd, res_baseline, res_student, w_diff_sgd, w_diff_baseline, w_diff_student, 0, self.opt.seed)

    def data_sampler(self, X, Y, i):
        i_min = i * self.opt.batch_size
        i_max = (i + 1) * self.opt.batch_size

        x = X[i_min:i_max].cuda()
        y = Y[i_min:i_max].cuda()

        return x, y

    def perceptual_loss(self, random_samples, selected_samples, generated_samples, feat_privacy_set, net, proj_matrix=None):
        self.opt.experiment = "PERCEPTUAL"
        print("Start training {} ...".format(self.opt.experiment))
        logname = os.path.join(self.opt.log_path, 'results' + '_' + self.opt.experiment + '_' + str(self.opt.seed) + '.csv')
        if not os.path.exists(logname):
            with open(logname, 'w') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow(['iter', 'sgd perceptual loss', 'baseline perceptual loss', 'student perceptual loss'])

        pdist = torch.nn.PairwiseDistance(p=2)

        unproj_matrix = torch.linalg.pinv(proj_matrix)

        loss = []
        img_shape = (1, 28, 28)
        experiments_lst = ['SGD', 'IMT_Baseline', 'Student']

        experiment_dict = {
            'SGD': [],
            'IMT_Baseline': [],
            'Student': []
        }

        for experiment in experiments_lst:
            if experiment == 'Student':
                for idx in tqdm(range(len(generated_samples))):
                    images = generated_samples[idx, :]
                    images = torch.from_numpy(images).unsqueeze(0).cuda()

                    # img = images.squeeze().cpu().numpy()
                    # plt.imshow(img, cmap="gray")

                    feat = net(images)
                    # tmp_score = cos(feat, feat_privacy_set)
                    tmp_score = pdist(feat, feat_privacy_set).min()
                    experiment_dict[experiment].append(tmp_score.item())
            elif experiment == "SGD":
                for idx in tqdm(range(len(random_samples))):
                    images = random_samples[idx, :]
                    images = torch.from_numpy(images).unsqueeze(0).cuda()

                    feat = net(images)
                    # tmp_score = cos(feat, feat_privacy_set)
                    tmp_score = pdist(feat, feat_privacy_set).min()
                    experiment_dict[experiment].append(tmp_score.item())
            else:
                for idx in tqdm(range(len(selected_samples))):
                    images = selected_samples[idx, :]
                    images = torch.from_numpy(images).unsqueeze(0).cuda()

                    feat = net(images)
                    # tmp_score = cos(feat, feat_privacy_set)
                    tmp_score = pdist(feat, feat_privacy_set).min()
                    experiment_dict[experiment].append(tmp_score.item())

        for t in range(len(generated_samples)):
            sgd_perceptual_loss = experiment_dict['SGD'][t]
            baseline_perceptual_loss = experiment_dict['IMT_Baseline'][t]
            student_perceptual_loss = experiment_dict['Student'][t]

            with open(logname, 'a') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow([t, sgd_perceptual_loss, baseline_perceptual_loss, student_perceptual_loss])

        return loss

