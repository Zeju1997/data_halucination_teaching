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
import torchvision.utils as vutils
from torchvision.utils import save_image, make_grid
from train_utils import *
import teachers.omniscient_teacher as omniscient
import teachers.utils as utils
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cv2

from datasets import BaseDataset

from experiments import SGDTrainer, IMTTrainer, WSTARTrainer

import networks.cgan as cgan
import networks.unrolled_vae as unrolled

from sklearn.datasets import make_moons, make_classification
from sklearn.model_selection import train_test_split

from utils.visualize import make_results_video, make_results_video_2d, make_results_img, make_results_img_2d, plot_generated_samples
from utils.data import init_data, load_experiment_result, plot_graphs
from utils.network import initialize_weights

from vaes.models import VAE_bMNIST, VAE_HalfMoon

import subprocess
import glob

import csv

sys.path.append('..') #Hack add ROOT DIR
from baseconfig import CONF




def plot_classifier(model, max, min):
    w = 0
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            w = layer.state_dict()['weight'].cpu().numpy()

    slope = (-w[0, 0]/w[0, 1] - 1) / (1 + w[0, 1]/w[0, 0])

    x = np.linspace(min, max, 100)
    y = slope * x
    return x, y


def to_matrix(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]


def to_img(x):
    x = x.clamp(0, 1)
    return x


def show_image(img, title):
    img = to_img(img)
    npimg = img.numpy()
    plt.title(title)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


class Trainer:
    def __init__(self, options):
        self.opt = options

        self.opt.model_name = "whitebox_unrolled_vae_" + self.opt.data_mode

        self.opt.log_path = os.path.join(CONF.PATH.LOG, self.opt.model_name)
        if not os.path.exists(self.opt.log_path):
            os.makedirs(self.opt.log_path)

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
            self.teacher.apply(initialize_weights)
            torch.save(self.teacher.state_dict(), 'teacher_w0.pth')
            # self.teacher.load_state_dict(torch.load('teacher_w0.pth'))

            self.student = omniscient.OmniscientLinearStudent(self.opt.dim)
            self.baseline = omniscient.OmniscientLinearStudent(self.opt.dim)

            # self.teacher = omniscient.TeacherClassifier(self.opt.dim)
            # self.student = omniscient.StudentClassifier(self.opt.dim)

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

        x = X[i_min:i_max].to(self.device)
        y = Y[i_min:i_max].to(self.device)

        return x, y

    def main(self):
        """Run the random teacher (SGD), baseline (IMT) and the DHT sequentially
        """

        print("Training")
        # self.set_train()

        if self.opt.init_data:
            init_data(self.opt)

        X = torch.load('X.pt')
        Y = torch.load('Y.pt')

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

            data_train = BaseDataset(X_train, Y_train)
            train_loader = DataLoader(data_train, batch_size=self.opt.batch_size, drop_last=True, shuffle=True)

        # data_train = BaseDataset(X_train, Y_train)
        # data_test = BaseDataset(X_test, Y_test)
        # train_loader = DataLoader(data_train, batch_size=self.opt.batch_size, drop_last=True)
        # test_loader = DataLoader(data_test, batch_size=self.opt.batch_size, drop_last=True)

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
        if self.opt.train_sgd == False:

            sgd_example = utils.BaseLinear(self.opt.dim)
            sgd_example.load_state_dict(torch.load('teacher_w0.pth'))

            sgd_trainer = SGDTrainer(self.opt, X_train, Y_train, X_test, Y_test)
            _, _ = sgd_trainer.train(sgd_example, w_star)

        res_sgd, w_diff_sgd = load_experiment_result(self.opt)

        # ---------------------
        #  Train IMT Baseline
        # ---------------------

        self.opt.experiment = "IMT_Baseline"
        if self.opt.train_baseline == False:
            self.baseline.load_state_dict(torch.load('teacher_w0.pth'))

            imt_trainer = IMTTrainer(self.opt, X_train, Y_train, X_test, Y_test)
            _, _ = imt_trainer.train(self.baseline, self.teacher, w_star)

        res_baseline, w_diff_baseline = load_experiment_result(self.opt)

        # ---------------------
        #  Train Student
        # ---------------------
        if self.visualize == False:
            vae = VAE_bMNIST(self.device)
            vae = vae.to(self.device)

            optimizer = torch.optim.Adam(params=vae.parameters(), lr=0.001, weight_decay=1e-5)

            # set to training mode
            vae.train()

            train_loss_avg = []

            print('Training ...')
            self.opt.n_epochs = 300
            for epoch in range(self.opt.n_epochs):
                train_loss_avg.append(0)
                num_batches = 0

                for x_batch, y_batch in train_loader:
                    optimizer.zero_grad()

                    y_batch = F.one_hot(y_batch.long(), num_classes=2).type(torch.FloatTensor) * 2. - 1
                    y_batch = y_batch.to(self.device)
                    x_batch = x_batch.to(self.device)

                    loss, _ = vae(x_batch, y_batch)

                    # backpropagation
                    loss.backward()

                    # one step of the optmizer (using the gradients from backpropagation)
                    optimizer.step()

                    train_loss_avg[-1] += loss.item()
                    num_batches += 1

                train_loss_avg[-1] /= num_batches
                print('Epoch [%d / %d] average negative ELBO: %f' % (epoch+1, self.opt.n_epochs, train_loss_avg[-1]))

            vae.eval()
            with torch.no_grad():

                # sample images
                img_samples, y_logits = vae.sample()
                y = torch.argmax(y_logits, dim=1).data.cpu().numpy()
                print("Samples:")
                print(y)
                img_samples = img_samples.cpu()

                fig, ax = plt.subplots(figsize=(5, 5))
                show_image(torchvision.utils.make_grid(img_samples,10,5), "Samples")
                plt.show()

            torch.save(vae.state_dict(), 'pretrained_vae.pth')

            sys.exit()

        if self.opt.train_student == True:
            self.opt.experiment = "Student"
            print("Start training {} ...".format(self.opt.experiment))
            logname = os.path.join(self.opt.log_path, 'results' + '_' + self.opt.experiment + '_' + str(self.opt.seed) + '.csv')
            if not os.path.exists(logname):
                with open(logname, 'w') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    logwriter.writerow(['iter', 'test acc', 'w diff'])

            tmp_student = utils.BaseLinear(self.opt.dim)

            if self.opt.data_mode == "mnist":
                netG = unrolled.Generator(self.opt, self.teacher, tmp_student).to(self.device)
                vae = VAE_bMNIST(self.device).to(self.device)
                unrolled_optimizer = unrolled.UnrolledOptimizer(opt=self.opt, teacher=self.teacher, student=tmp_student, generator=netG, vae=vae, X=X_train.to(self.device), Y=Y_train.to(self.device), proj_matrix=proj_matrix)
            else:
                netG = unrolled.Generator_moon(self.opt, self.teacher, tmp_student).to(self.device)
                vae = VAE_bMNIST(self.device).to(self.device)
                unrolled_optimizer = unrolled.UnrolledOptimizer_moon(opt=self.opt, teacher=self.teacher, student=tmp_student, generator=netG, vae=vae, X=X_train.to(self.device), Y=Y_train.to(self.device))

            netG.apply(initialize_weights)
            optimG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

            self.step = 0
            loss_student = []
            img_shape = (1, 28, 28)
            w_init = self.student.lin.weight

            for epoch in tqdm(range(self.opt.n_epochs)):
                if epoch != 0:
                    for i, (data, labels) in enumerate(train_loader):
                        self.step = self.step + 1

                        # -----------------
                        #  Train Generator
                        # -----------------

                        optimG.zero_grad()

                        w_t = netG.state_dict()
                        gradients, generator_loss = unrolled_optimizer(w_t, w_star)

                        loss_student.append(generator_loss.item())

                        with torch.no_grad():
                            for p, g in zip(netG.parameters(), gradients):
                                p.grad = g

                        optimG.step()

                        print("{}/{}".format(i, len(train_loader)))

                    # fig = plt.figure()
                    # plt.plot(loss_student, c="b", label="Teacher (CNN)")
                    # plt.xlabel("Epoch")
                    # plt.ylabel("Accuracy")
                    # plt.legend()
                    # plt.show()

            res_student = []
            a_student = []
            b_student = []
            w_diff_student = []

            self.student.load_state_dict(torch.load('teacher_w0.pth'))

            generated_samples = np.zeros(2)
            for idx in tqdm(range(self.opt.n_iter)):
                if idx != 0:
                    w_t = self.student.lin.weight
                    w_t = w_t / torch.norm(w_t)

                    i = torch.randint(0, nb_batch, size=(1,)).item()
                    i_min = i * self.opt.batch_size
                    i_max = (i + 1) * self.opt.batch_size

                    gt_x = X_train[i_min:i_max].to(self.device)
                    y = Y_train[i_min:i_max].to(self.device)

                    z = Variable(torch.randn((self.opt.batch_size, self.opt.latent_dim))).cuda()
                    w = torch.cat((w_t, w_t-w_star), dim=1)
                    w = w.repeat(self.opt.batch_size, 1)
                    x = torch.cat((w, gt_x), dim=1)

                    z, qz_mu, qz_std = netG(x, y)
                    generated_sample, y_logit = vae.p_xy(z)

                    if idx == 1:
                        generated_samples = generated_sample.cpu().detach().numpy()  # [np.newaxis, :]
                        generated_labels = y.cpu().detach().numpy()  # [np.newaxis, :]
                    else:
                        generated_samples = np.concatenate((generated_samples, generated_sample.cpu().detach().numpy()), axis=0)
                        generated_labels = np.concatenate((generated_labels, y.cpu().detach().numpy()), axis=0)

                    generated_sample = generated_sample.view(self.opt.batch_size, -1)
                    generated_sample = generated_sample @ proj_matrix.to(self.device)
                    self.student.update(generated_sample.detach(), y.unsqueeze(1))

                self.student.eval()
                test = self.student(X_test.to(self.device)).cpu()

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

            if self.opt.data_mode == "moon":
                make_results_img_2d(self.opt, X, Y, generated_samples, generated_labels, res_sgd, res_baseline, res_student, w_diff_sgd, w_diff_baseline, w_diff_student, 0, self.opt.seed)
                # make_results_video_2d(self.opt, X, Y, generated_samples, generated_labels, res_sgd, res_baseline, res_student, w_diff_sgd, w_diff_baseline, w_diff_student, epoch, self.opt.seed)

            else:
                # make_results_img(self.opt, X, Y, generated_samples, generated_labels, res_sgd, res_baseline, res_student, w_diff_sgd, w_diff_baseline, w_diff_student, 0, self.opt.seed)
                # make_results_video(self.opt, X, Y, generated_samples, generated_labels, res_sgd, res_baseline, res_student, w_diff_sgd, w_diff_baseline, w_diff_student, 0, self.opt.seed)
                plot_generated_samples(self.opt, X, Y, generated_samples, generated_labels, epoch, self.opt.seed)

            save_folder = os.path.join(self.opt.log_path, "models", "weights_{}".format(epoch))
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            save_path = os.path.join(save_folder, "netG_{}.pth".format("models", epoch))
            to_save = netG.state_dict()
            torch.save(to_save, save_path)

            # self.make_results_video_generated_data(generated_samples, epoch)

    def plot_results(self):

        experiments_lst = ['SGD', 'IMT_Baseline', 'Student']
        rootdir = self.opt.log_path

        experiment_dict = {
            'SGD': [],
            'IMT_Baseline': [],
            'Student': []
        }

        for experiment in experiments_lst:
            for file in os.listdir(rootdir):
                if file.endswith('.csv'):
                    if experiment in file:
                        experiment_dict[experiment].append(file)

        plot_graphs(rootdir, experiment_dict, experiments_lst)