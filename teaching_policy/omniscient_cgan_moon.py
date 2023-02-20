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
import cv2

from datasets import BaseDataset

from experiments import SGDTrainer, IMTTrainer, WSTARTrainer

import networks.cgan as cgan
import networks.unrolled_cgan as unrolled


from utils.visualize import make_results_video, make_results_video_2d, make_results_img, make_results_img_2d, plot_generated_samples_2d, plot_classifier, plot_distribution
from utils.data import init_data, plot_graphs, load_experiment_result
from utils.network import initialize_weights

import subprocess
import glob

import csv

sys.path.append('..') #Hack add ROOT DIR
from baseconfig import CONF



class Trainer:
    def __init__(self, options):
        self.opt = options

        self.opt.model_name = "omniscient_cgan_" + self.opt.data_mode

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

        if self.opt.train_student == True:
            self.opt.experiment = "Student"
            print("Start training {} ...".format(self.opt.experiment))
            logname = os.path.join(self.opt.log_path, 'results' + '_' + self.opt.experiment + '_' + str(self.opt.seed) + '.csv')
            if not os.path.exists(logname):
                with open(logname, 'w') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    logwriter.writerow(['iter', 'test acc', 'w diff'])

            adversarial_loss = torch.nn.BCELoss()

            tmp_student = utils.BaseLinear(self.opt.dim)

            if self.opt.data_mode == "mnist":
                netG = unrolled.Generator(self.opt, self.teacher, tmp_student).cuda()
                netD = unrolled.Discriminator(self.opt).cuda()
                unrolled_optimizer = unrolled.UnrolledOptimizer(opt=self.opt, teacher=self.teacher, student=tmp_student, generator=netG, X=X_train.cuda(), Y=Y_train.cuda(), proj_matrix=proj_matrix)
            else:
                netG = unrolled.Generator_moon(self.opt, self.teacher, tmp_student).cuda()
                netD = unrolled.Discriminator_moon(self.opt).cuda()
                unrolled_optimizer = unrolled.UnrolledOptimizer_moon(opt=self.opt, teacher=self.teacher, student=tmp_student, generator=netG, X=X_train.cuda(), Y=Y_train.cuda())

            netG.apply(initialize_weights)
            netD.apply(initialize_weights)

            optimD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
            optimG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

            self.step = 0
            loss_student = []
            img_shape = (1, 28, 28)
            w_init = self.student.lin.weight

            for epoch in tqdm(range(self.opt.n_epochs)):
                if epoch != 0:
                    for i, (data, labels) in enumerate(train_loader):
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

                        optimG.zero_grad()

                        w_t = netG.state_dict()
                        gradients, generator_loss, g_loss, validity, generated_samples, generated_labels = unrolled_optimizer(w_t, w_star, netD, valid)

                        loss_student.append(generator_loss.item())

                        with torch.no_grad():
                            for p, g in zip(netG.parameters(), gradients):
                                p.grad = g

                        optimG.step()

                        # ---------------------
                        #  Train Discriminator
                        # ---------------------

                        for _ in range(self.opt.n_critic):
                            optimD.zero_grad()

                            # Loss for real images
                            validity_real = netD(real_samples, real_labels)
                            d_real_loss = adversarial_loss(validity_real, valid)

                            # Loss for fake images
                            validity_fake = netD(generated_samples.detach(), Variable(generated_labels.type(torch.cuda.LongTensor)))
                            d_fake_loss = adversarial_loss(validity_fake, fake)

                            # Total discriminator loss
                            d_loss = (d_real_loss + d_fake_loss) / 2

                            d_loss.backward()
                            optimD.step()

                        if i % self.opt.log_frequency == 0:
                            print(
                                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                                % (epoch, self.opt.n_epochs, i, len(train_loader), d_loss.item(), g_loss.item())
                            )
                            self.log("train", d_loss.item(), g_loss.item(), self.step)

            res_student = []
            a_student = []
            b_student = []
            w_diff_student = []

            self.student.load_state_dict(torch.load('teacher_w0.pth'))

            generated_samples = np.zeros(2)
            for idx in tqdm(range(self.opt.n_iter)):
                if idx != 0:
                    w_t = self.student.lin.weight

                    i = torch.randint(0, nb_batch, size=(1,)).item()
                    gt_x, gt_y = self.data_sampler(X_train, Y_train, i)

                    # z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, gt_x.shape)))
                    z = Variable(torch.randn((self.opt.batch_size, self.opt.latent_dim))).cuda()

                    # x = torch.cat((w_t, w_t-w_star, gt_x, y.unsqueeze(0)), dim=1)
                    x = torch.cat((w_t, w_t-w_star, gt_x), dim=1)
                    generated_sample = netG(x, gt_y)

                    if idx == 1:
                        generated_samples = generated_sample.cpu().detach().numpy()  # [np.newaxis, :]
                        generated_labels = gt_y.unsqueeze(1).cpu().detach().numpy()  # [np.newaxis, :]
                    else:
                        generated_samples = np.concatenate((generated_samples, generated_sample.cpu().detach().numpy()), axis=0)
                        generated_labels = np.concatenate((generated_labels, gt_y.unsqueeze(1).cpu().detach().numpy()), axis=0)

                    # generated_sample = generated_sample @ proj_matrix.cuda()
                    self.student.update(generated_sample.detach(), gt_y.unsqueeze(1))

                self.student.eval()
                test = self.student(X_test.cuda()).cpu()

                a, b = plot_classifier(self.student, X_test[:, 0].max(axis=0), X_test[:, 0].min(axis=0))
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

            if self.opt.data_mode == "gaussian" or self.opt.data_mode == "moon":
                # make_results_img_2d(self.opt, X, Y, generated_samples, generated_labels, res_sgd, res_baseline, res_student, w_diff_sgd, w_diff_baseline, w_diff_student, epoch, self.opt.seed)
                # make_results_video_2d(self.opt, X, Y, generated_samples, generated_labels, res_sgd, res_baseline, res_student, w_diff_sgd, w_diff_baseline, w_diff_student, epoch, self.opt.seed)

                a_star, b_star = plot_classifier(self.teacher, X_test[:, 0].max(axis=0), X_test[:, 0].min(axis=0))
                plot_generated_samples_2d(self.opt, X, Y, a_star, b_star, a_student, b_student, generated_samples, generated_labels, epoch, self.opt.seed)

                plot_distribution(self.opt, X_train, Y_train, generated_samples, generated_labels)
            else:
                make_results_img(self.opt, X, Y, generated_samples, generated_labels, res_sgd, res_baseline, res_student, w_diff_sgd, w_diff_baseline, w_diff_student, epoch, self.opt.seed, proj_matrix)
                # make_results_video(self.opt, X, Y, generated_samples, generated_labels, res_sgd, res_baseline, res_student, w_diff_sgd, w_diff_baseline, w_diff_student, epoch, self.opt.seed, proj_matrix)

            save_folder = os.path.join(self.opt.log_path, "models", "weights_{}".format(epoch))
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            save_path = os.path.join(save_folder, "netG_{}.pth".format("models", epoch))
            to_save = netG.state_dict()
            torch.save(to_save, save_path)

            save_path = os.path.join(save_folder, "netD_{}.pth".format("models", epoch))
            to_save = netD.state_dict()
            torch.save(to_save, save_path)

            # self.make_results_video_generated_data(generated_samples, epoch)

    def load_experiment_result(self):
        """If already trained before, load the experiment results from the corresponding .csv file.
        """
        csv_path = os.path.join(self.opt.log_path, 'results' + '_' + self.opt.experiment + '_' + str(self.opt.seed) + '.csv')

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

    def compute_gradient_penalty(self, D, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = Variable(torch.Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def log(self, mode, name, value, step):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        writer.add_scalar("{}/{}/{}".format(self.opt.experiment, mode, name), value, step)

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
