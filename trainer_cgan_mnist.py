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
from torchvision.transforms import ToTensor
import torchvision.utils as vutils
from torchvision.utils import save_image, make_grid
from train_utils import *
from eval import EvalMetrics
import teachers.omniscient_teacher as omniscient
import teachers.utils as utils
import matplotlib.pyplot as plt
import cv2

import csv

from datasets import BaseDataset

import networks.cgan as cgan
import networks.unrolled_cgan as unrolled

from sklearn.datasets import make_moons, make_classification
from sklearn.model_selection import train_test_split

from utils.visualize import make_results_video, make_results_video_2d, make_results_img, make_results_img_2d
from utils.data import init_data

from experiments import SGDTrainer, IMTTrainer, WSTARTrainer

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

        self.opt.model_name = "whitebox_unrolled_cgan_" + self.opt.data_mode

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
            torch.save(self.teacher.state_dict(), 'teacher_w0.pth')
            # self.teacher.load_state_dict(torch.load('teacher_w0.pth'))

            self.student = omniscient.OmniscientLinearStudent(self.opt.dim)
            self.baseline = omniscient.OmniscientLinearStudent(self.opt.dim)

            # self.teacher = omniscient.TeacherClassifier(self.opt.dim)
            # self.student = omniscient.StudentClassifier(self.opt.dim)

        self.student.load_state_dict(torch.load('teacher_w0.pth'))
        self.baseline.load_state_dict(torch.load('teacher_w0.pth'))

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

        if self.opt.init_data:
            init_data(self.opt)

        X = torch.load('X.pt')
        Y = torch.load('Y.pt')
        proj_matrix = torch.load('proj_matrix.pt')

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
        netG = unrolled.Generator(self.opt).cuda()
        netG.apply(weights_init)

        netD = unrolled.Discriminator(self.opt).cuda()
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

        if self.opt.train_sgd == True:

            sgd_example = utils.BaseLinear(self.opt.dim)
            sgd_example.load_state_dict(torch.load('teacher_w0.pth'))

            sgd_trainer = SGDTrainer(self.opt, X_train, Y_train, X_test, Y_test)
            sgd_trainer.train(sgd_example, w_star)

        self.experiment = "SGD"
        res_sgd, w_diff_sgd = self.load_experiment_result()

        # ---------------------
        #  Train IMT Baseline
        # ---------------------

        if self.opt.train_baseline == True:
            self.baseline.load_state_dict(torch.load('teacher_w0.pth'))

            imt_trainer = IMTTrainer(self.opt, X_train, Y_train, X_test, Y_test)
            imt_trainer.train(self.baseline, self.teacher, w_star)

        self.experiment = "IMT_Baseline"
        res_baseline, w_diff_baseline = self.load_experiment_result()

        # ---------------------
        #  Train Student
        # ---------------------

        if self.opt.train_student == True:
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

            indices = np.random.choice(X_train.shape[0], self.opt.n_classes**2)
            x_test = X_train[indices].cuda()
            x_test = x_test / torch.norm(x_test)

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
                gen_losses = 0

                if epoch != 0:
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
                        gradients, generator_loss, G_loss, z_out, generated_samples = unrolled_optimizer(w_t, w_star, netD, generated_labels, real, epoch)
                        # loss_student.append(generator_loss.item())
                        gen_losses = gen_losses + generator_loss.item() / nb_batch

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

                        for _ in range(self.opt.n_critic):
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

                    loss_student.append(gen_losses)

                    # plt.plot(loss_student, c='b', label="loss")
                    # plt.title(str(self.opt.data_mode) + "Model (class : " + str(self.opt.class_1) + ", " + str(self.opt.class_2) + ")")
                    # plt.xlabel("Iteration")
                    # plt.ylabel("Loss")
                    # plt.legend()
                    # plt.show()

                    # calculate average value for one epoch
                    D_losses.append(sum(epoch_D_losses)/len(epoch_D_losses))
                    G_losses.append(sum(epoch_G_losses)/len(epoch_G_losses))
                    Dx_values.append(sum(epoch_Dx)/len(epoch_Dx))
                    DGz_values.append(sum(epoch_DGz)/len(epoch_DGz))

                    print(f" Epoch {epoch+1}/{self.opt.n_epochs} Discriminator Loss {D_losses[-1]:.3f} Generator Loss {G_losses[-1]:.3f}"
                         + f" D(x) {Dx_values[-1]:.3f} D(G(x)) {DGz_values[-1]:.3f}")

                if epoch % self.opt.save_frequency == 0 and epoch >= 1:
                    res_student = []
                    a_student = []
                    b_student = []
                    w_diff_student = []

                    self.student.load_state_dict(torch.load('teacher_w0.pth'))
                    netG.eval()
                    for idx in tqdm(range(self.opt.n_iter)):
                        if idx != 0:
                            w_t = self.student.lin.weight
                            w_t = w_t / torch.norm(w_t)

                            i = torch.randint(0, nb_batch, size=(1,)).item()
                            gt_x, gt_y = self.data_sampler(X_train, Y_train, i)
                            gt_y_onehot = onehot[gt_y.long()].cuda()
                            gt_x = gt_x / torch.norm(gt_x)

                            # z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, gt_x.shape)))
                            # z = Variable(torch.randn((self.opt.batch_size, self.opt.latent_dim-))).cuda()

                            # w = torch.cat((w_t, w_t-w_star), dim=1).repeat(self.opt.batch_size, 1)
                            z = torch.randn(self.opt.batch_size, self.opt.latent_dim).cuda()
                            w = torch.cat((w_t, w_t-w_star), dim=1)
                            w = w.repeat(self.opt.batch_size, 1)
                            # x = torch.cat((w, z), dim=1)
                            generated_sample = netG(w, gt_y_onehot)

                            if idx == 1:
                                generated_samples = generated_sample.cpu().detach().numpy()  # [np.newaxis, :]
                                generated_labels = gt_y.cpu().detach().numpy()  # [np.newaxis, :]
                            else:
                                generated_samples = np.concatenate((generated_samples, generated_sample.cpu().detach().numpy()), axis=0)
                                generated_labels = np.concatenate((generated_labels, gt_y.cpu().detach().numpy()), axis=0)

                            if self.opt.data_mode == "mnist":
                                generated_sample = generated_sample.reshape((self.opt.batch_size, self.opt.img_size**2))
                                generated_sample = generated_sample.detach().clone() @ proj_matrix.cuda()

                            self.student.update(generated_sample, gt_y)

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
                        w = self.student.lin.weight
                        w = w / torch.norm(w)
                        diff = torch.linalg.norm(w_star - w, ord=2) ** 2
                        w_diff_student.append(diff.detach().clone().cpu())

                    '''
                    if self.opt.data_mode == "mnist":
                        save_folder = os.path.join(self.opt.log_path, "imgs")
                        if not os.path.exists(save_folder):
                            os.makedirs(save_folder)
    
                        img_path = os.path.join(save_folder, "results_examples_{}.png".format(epoch))
    
                        netG.eval()
                        with torch.no_grad():
                            w_t = self.student.lin.weight
                            w_t = w_t / torch.norm(w_t)
    
                            print("wt", w_t.mean())
    
                            w = torch.cat((w_t, w_t-w_star), dim=1)
                            w = w.repeat(self.opt.n_classes**2, 1)
                            x = torch.cat((w, x_test), dim=1)
                            fake_test = netG(w, test_labels_onehot).cpu()
                            torchvision.utils.save_image(fake_test, img_path, nrow=10, padding=0, normalize=True)
                        netG.train()
                    '''

                    if self.opt.data_mode == "gaussian" or self.opt.data_mode == "moon":
                        make_results_img_2d(self.opt, X, Y, a_student, b_student, generated_samples, generated_labels, res_sgd, res_baseline, res_student, w_diff_sgd, w_diff_baseline, w_diff_student, epoch)
                        make_results_video_2d(self.opt, X, Y, a_student, b_student, generated_samples, generated_labels, res_sgd, res_baseline, res_student, w_diff_sgd, w_diff_baseline, w_diff_student, epoch)
                    else:
                        make_results_img(self.opt, X, Y, a_student, b_student, generated_samples, generated_labels, res_sgd, res_baseline, res_student, w_diff_sgd, w_diff_baseline, w_diff_student, epoch)
                        make_results_video(self.opt, X, Y, a_student, b_student, generated_samples, generated_labels, res_sgd, res_baseline, res_student, w_diff_sgd, w_diff_baseline, w_diff_student, epoch)

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

            '''
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
            '''

            if self.visualize == True:
                fig, (ax1, ax2) = plt.subplots(1, 2)
                fig.set_size_inches(12, 6)
                ax1.plot(res_sgd, c='g', label="SGD %s" % self.opt.data_mode)
                ax1.plot(res_baseline, c='b', label="IMT %s" % self.opt.data_mode)
                ax1.plot(res_student, c='r', label="Student %s" % self.opt.data_mode)
                # ax1.axhline(y=teacher_acc, color='k', linestyle='-', label="teacher accuracy")
                ax1.set_title("Test accuracy " + str(self.opt.data_mode) + " (class : " + str(self.opt.class_1) + ", " + str(self.opt.class_2) + ")")
                ax1.set_xlabel("Iteration")
                ax1.set_ylabel("Accuracy")
                ax1.legend(loc="lower right")

                ax2.plot(w_diff_sgd, 'go', label="SGD %s" % self.opt.data_mode)
                ax2.plot(w_diff_baseline, 'bo', label="IMT %s" % self.opt.data_mode, alpha=0.5)
                ax2.plot(w_diff_student, 'ro', label="Student %s" % self.opt.data_mode, alpha=0.5)
                ax2.legend(loc="lower left")
                ax2.set_title("w diff " + str(self.opt.data_mode) + " (class : " + str(self.opt.class_1) + ", " + str(self.opt.class_2) + ")")
                ax2.set_xlabel("Iteration")
                ax2.set_ylabel("Distance between $w^t$ and $w^*$")
                #ax2.set_aspect('equal')

                # plt.savefig('results_mnist_final.jpg')
                # plt.close()
                plt.show()

    def load_experiment_result(self):
        """Write an event to the tensorboard events file
        """
        csv_path = os.path.join(self.opt.log_path, 'results' + '_' + self.experiment + '_' + str(self.opt.seed) + '.csv')

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

    def make_results_video_generated_data(self, generated_samples, epoch):
        n = generated_samples.shape[0]
        for i in tqdm(range(n)):
            img_path = self.opt.log_path + "/file%03d.png" % i
            img = generated_samples[i]
            img = torch.from_numpy(img)
            torchvision.utils.save_image(img, img_path, nrow=10, padding=0, normalize=True)

        #'-vf', 'scale=320:320'
        subprocess.call([
            'ffmpeg', '-framerate', '8', '-i', self.opt.log_path + '/file%03d.png', '-r', '30', '-pix_fmt', 'yuv420p',
            self.opt.log_path + '/generated_samples_epoch_{}.mp4'.format(epoch)
        ])
        for file_name in glob.glob(self.opt.log_path + '/' + "*.png"):
            os.remove(file_name)


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


    def main1(self):
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