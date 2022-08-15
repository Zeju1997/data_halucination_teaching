from __future__ import absolute_import, division, print_function

import numpy as np
import time
import json

import sys

import csv

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
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
from torchvision.utils import save_image, make_grid
from train_utils import *
from eval import EvalMetrics
import teachers.omniscient_teacher as omniscient
import teachers.surrogate_teacher as surrogate
import teachers.imitation_teacher as imitation
import teachers.utils as utils
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

from datasets import MoonDataset

from datasets import BaseDataset

import networks.cgan as cgan
import networks.unrolled_optimizer as unrolled
import networks.blackbox_mixup_cnn as blackbox_mixup
# import networks

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
        torch.nn.init.xavier_uniform_(m.weight)
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


def mixup_data1(gt_x_1, gt_x_2, gt_y_1, gt_y_2, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''

    # batch_size = gt_x.size()[0]
    # index = torch.randperm(batch_size).cuda()

    # mixed_x = lam * gt_x + (1 - lam) * gt_x[index, :]
    # y_a, y_b = gt_y, gt_y[index]

    y_a = gt_y_1
    y_b = gt_y_2
    mixed_x = alpha * gt_x_1 + (1 - alpha) * gt_x_2

    return mixed_x, y_a, y_b


def mixup_criterion1(criterion, pred, y_a, y_b, lam):
    loss = lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    return loss.to(torch.float32)


def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    batch_size = x.shape[0]
    index = torch.randperm(batch_size).cuda()

    if alpha != 1:
        lam = np.random.beta(alpha, alpha, size=(x.shape[0]))
        lam = torch.tensor(lam, dtype=torch.float).cuda()
        # mixed_y = lam * y + (1 - lam) * y[index]

        lam = torch.unsqueeze(lam, 1)
        lam = torch.unsqueeze(lam, 2)
        lam = torch.unsqueeze(lam, 3)
        mixed_x = lam * x + (1 - lam) * x[index, :]
    else:
        lam = np.random.beta(alpha, alpha)
        mixed_x = lam * x + (1 - lam) * x[index, :]

    y_a, y_b = y, y[index]
    # mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    loss = lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    return loss


def mixup_criterion_batch(criterion, pred, y_a, y_b, lam):
    loss = lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    loss = torch.mean(loss)
    return loss

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


def weight_diff(w_star, w):
    diff = 0
    for i in range(len(w_star)):
        # w = w_star[i] - w[i]
        # diff = diff + torch.linalg.norm(w_star[i] - w[i], dim=(0, 1, 2, 3), ord=2) ** 2
        diff = diff + torch.linalg.norm(w_star[i] - w[i]) ** 2
    return diff

def initialize_weights(m):
  if isinstance(m, nn.Conv2d):
      nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
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

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.opt.log_path = os.path.join(CONF.PATH.LOG, self.opt.model_name)

        self.visualize = True

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        if self.opt.data_mode == "cifar10":
            if self.opt.augment:
                transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010)),
                ])
            else:
                transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010)),
                ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ])
            self.train_dataset = torchvision.datasets.CIFAR10(root=CONF.PATH.DATA, train=True, download=True, transform=transform_train)
            self.test_dataset = torchvision.datasets.CIFAR10(root=CONF.PATH.DATA, train=False, download=True, transform=transform_test)
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.opt.batch_size)
            # self.train_loader = DataLoader(self.train_dataset, batch_size=len(self.train_dataset))
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.opt.batch_size)
            # self.test_loader = DataLoader(self.test_dataset, batch_size=len(self.test_dataset))
        elif self.opt.data_mode == "cifar100":
            if self.opt.augment:
                transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4865, 0.4409),
                                         (0.2673, 0.2564, 0.2762)),
                ])
            else:
                transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4865, 0.4409),
                                         (0.2673, 0.2564, 0.2762)),
                ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409),
                                     (0.2673, 0.2564, 0.2762)),
            ])
            self.train_dataset = torchvision.datasets.CIFAR100(root=CONF.PATH.DATA, train=True, download=True, transform=transform_train)
            self.test_dataset = torchvision.datasets.CIFAR100(root=CONF.PATH.DATA, train=False, download=True, transform=transform_test)
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.opt.batch_size)
            # self.train_loader = DataLoader(self.train_dataset, batch_size=len(self.train_dataset))
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.opt.batch_size)
            # self.test_loader = DataLoader(self.test_dataset, batch_size=len(self.test_dataset))
        elif self.opt.data_mode == "mnist":
            # MNIST normalizing
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
            self.train_dataset = torchvision.datasets.MNIST(root=CONF.PATH.DATA, train=True, download=True, transform=transform)
            # train, valid = random_split(train_dataset, [50000, 10000])
            self.test_dataset = torchvision.datasets.MNIST(root=CONF.PATH.DATA, train=False, download=True, transform=transform)
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.opt.batch_size, shuffle=True)
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.opt.batch_size, shuffle=True)
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
        for mode in ["train", "test"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.opt.log_path, mode))

        # self.loss_fn = nn.MSELoss()
        self.loss_fn = nn.CrossEntropyLoss()

        self.step = 0
        self.best_acc = 0
        self.best_test_loss = 0
        self.init_train_loss = 0
        self.init_test_loss = 0

        self.experiment = "teacher"

    def get_teacher_student(self):
        if self.opt.teaching_mode == "omniscient":
            if self.opt.data_mode == "cifar10":
                self.teacher = networks.ResNet18(in_channels=3).cuda()
                torch.save(self.teacher.state_dict(), 'teacher_w0.pth')
                # self.teacher.load_state_dict(torch.load('teacher.pth'))

                self.student = networks.ResNet18(in_channels=3).cuda()
                self.baseline = networks.ResNet18(in_channels=3).cuda()
            elif self.opt.data_mode == "cifar100":
                # self.teacher = networks.ResNet18(in_channels=3, num_classes=100).cuda()
                self.teacher = networks.CNN(in_channels=3, num_classes=100).cuda()
                self.teacher.apply(initialize_weights)
                torch.save(self.teacher.state_dict(), 'teacher_w0.pth')
                # self.teacher.load_state_dict(torch.load('teacher.pth'))

                # self.student = networks.ResNet18(in_channels=3, num_classes=100).cuda()
                self.student = networks.CNN(in_channels=3, num_classes=100).cuda()
                # self.baseline = networks.ResNet18(in_channels=3, num_classes=100).cuda()
                self.baseline = networks.CNN(in_channels=3, num_classes=100).cuda()
            else: # mnist / gaussian / moon
                self.teacher = networks.ResNet18(in_channels=1).cuda()
                torch.save(self.teacher.state_dict(), 'teacher_w0.pth')
                # self.teacher.load_state_dict(torch.load('teacher.pth'))

                self.student = networks.ResNet18(in_channels=1).cuda()
                self.baseline = networks.ResNet18(in_channels=1).cuda()

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

    def adjust_learning_rate(self, optimizer, epoch):
        """decrease the learning rate at 100 and 150 epoch"""
        lr = self.opt.lr
        if epoch >= 100:
            lr /= 10
        if epoch >= 150:
            lr /= 10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def main(self):
        """Run a single epoch of training and validation
        """

        print("Training")
        # self.set_train()

        torch.manual_seed(self.opt.seed)
        # np.random.seed(args.seed)
        # torch.cuda.set_device(args.gpu)
        # cudnn.benchmark = True
        # torch.manual_seed(args.seed)
        # cudnn.enabled=True
        # torch.cuda.manual_seed(args.seed)

        if self.opt.data_mode == "cifar10":
            example = networks.ResNet18(in_channels=3, num_classes=10).cuda()
            tmp_student = networks.ResNet18(in_channels=3, num_classes=10).cuda()
            mixup_baseline = networks.ResNet18(in_channels=3, num_classes=10).cuda()

        elif self.opt.data_mode == "cifar100":
            # example = networks.ResNet18(in_channels=3, num_classes=100).cuda()
            example = networks.CNN(in_channels=3, num_classes=100).cuda()
            # tmp_student = networks.ResNet18(in_channels=3, num_classes=100).cuda()
            tmp_student = networks.CNN(in_channels=3, num_classes=100).cuda()
            # mixup_baseline = networks.ResNet18(in_channels=3, num_classes=100).cuda()
            mixup_baseline = networks.CNN(in_channels=3, num_classes=100).cuda()

        elif self.opt.data_mode == "mnist":
            example = networks.ResNet18(in_channels=1, num_classes=10).cuda()
            tmp_student = networks.ResNet18(in_channels=1, num_classes=10).cuda()
            mixup_baseline = networks.ResNet18(in_channels=1, num_classes=10).cuda()

        else:
            print("Unrecognized data mode!")
            sys.exit()

        example.load_state_dict(self.teacher.state_dict())
        tmp_student.load_state_dict(self.teacher.state_dict())
        mixup_baseline.load_state_dict(self.teacher.state_dict())





        # train student
        netG = blackbox_mixup.Generator(self.opt).cuda()
        netG.apply(weights_init)

        optimizer_G = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999), eps=1e-08, amsgrad=False)
        # optimizer_G = torch.optim.SGD(netG.parameters(), lr=0.0002)
        unrolled_optimizer = blackbox_mixup.UnrolledBlackBoxOptimizer(opt=self.opt, teacher=self.teacher, student=tmp_student, generator=netG, train_dataset=self.train_dataset, proj_matrix=None)
        adversarial_loss = torch.nn.MSELoss()
        res_student = []
        res_loss_student = []
        a_student = []
        b_student = []
        loss_student = []
        cls_loss = []
        loss_g = []
        loss_d = []
        w_diff_student = []
        # w, h = generator.linear.weight.shape

        w_init = self.student.state_dict()
        # new_weight = w_init
        train_loss = []
        for idx in tqdm(range(self.opt.n_unroll)):
            if idx != 0:

                w_t = netG.state_dict()

                # w_student = []
                # for param in netG.parameters():
                #    w_student.append(param.data.clone())

                gradients, avg_loss, student_loss = unrolled_optimizer(w_t, w_init=w_init)

                print("idx", idx, "avg loss:", avg_loss)

                # if avg_loss < 5:
                    # torch.save(netG.state_dict(), 'netG.pth')
                    #break

                cls_loss = cls_loss + student_loss
                train_loss.append(avg_loss)

                optimizer_G.zero_grad()

                with torch.no_grad():
                    for p, g in zip(netG.parameters(), gradients):
                        p.grad = g

                optimizer_G.step()

                # w_g = []
                # for param in netG.parameters():
                #    w_g.append(param.data.clone())

                # print("asdlfjklasd")

                '''
                if idx % 100000 == 0:

                    fig = plt.figure()
                    plt.plot(loss_student, c="b", label="Teacher (CNN)")
                    plt.xlabel("Epoch")
                    plt.ylabel("Accuracy")
                    plt.legend()
                    plt.show()


                # cls_loss = cls_loss + loss_tmp
                cls_loss = loss_tmp

                loss_student.append(loss.item())

                with torch.no_grad():
                    for p, g in zip(netG.parameters(), gradients):
                        p.grad = g

                optimizer_G.step()
                '''
                ''''
                i = torch.randint(0, nb_batch, size=(1,)).item()
                gt_x_1, gt_y_1 = self.data_sampler(X_train, Y_train, i)

                i = torch.randint(0, nb_batch, size=(1,)).item()
                gt_x_2, gt_y_2 = self.data_sampler(X_train, Y_train, i)

                x = torch.cat((gt_x_1, gt_x_2), dim=1)
                alpha = netG(x, gt_y_1.long(), gt_y_2.long())

                mixed_x = alpha * gt_x_1 + (1 - alpha) * gt_x_2
                # mixed_y = alpha * gt_y_1 + (1 - alpha) * gt_y_2

                optimizer_G.zero_grad()
                out = self.student(mixed_x)
                # loss = mixup_criterion(self.loss_fn, out, targets_a.float(), targets_b.float(), alpha)
                loss = alpha * self.loss_fn(out, gt_y_1.long()) + (1 - alpha) * self.loss_fn(out, gt_y_2.long())
                loss = loss.to(torch.float32)

                # grad = torch.autograd.grad(loss, alpha, create_graph=True, retain_graph=True)
                # grad = torch.autograd.grad(alpha, model_paramters, create_graph=True, retain_graph=True)
                # grad = torch.autograd.grad(loss, model_paramters, create_graph=True, retain_graph=True)

                loss.backward() # create_graph=True, retain_graph=True)
                # loss.backward()
                optimizer_G.step()

                # i = torch.randint(0, nb_batch, size=(1,)).item()
                # gt_x_1, gt_y_1 = self.data_sampler(X_train, Y_train, i)

                # i = torch.randint(0, nb_batch, size=(1,)).item()
                # gt_x_2, gt_y_2 = self.data_sampler(X_train, Y_train, i)

                # x = torch.cat((gt_x_1, gt_x_2), dim=1)
                alpha = netG(x, gt_y_1.long(), gt_y_2.long())

                mixed_x = alpha * gt_x_1 + (1 - alpha) * gt_x_2

                student_optim.zero_grad()

                out = self.student(mixed_x)
                # loss = mixup_criterion(self.loss_fn, out, targets_a.float(), targets_b.float(), alpha)
                loss = alpha * self.loss_fn(out, gt_y_1.long()) + (1 - alpha) * self.loss_fn(out, gt_y_2.long())
                loss = loss.to(torch.float32)
                loss.backward()

                student_optim.step()

                train_loss.append(loss.item())
                '''

        if self.visualize == True:
            fig = plt.figure()
            plt.plot(cls_loss, c="b", label="Teacher (CNN)")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.show()
            # plt.close()

            fig = plt.figure()
            plt.plot(train_loss, c="r", label="Teacher (CNN)")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.show()
            # plt.close()

        sys.exit()


        # w_student1 = []
        # for param in self.student.parameters():
        #     w_student1.append(param.data.clone())

        # student_optim = torch.optim.SGD(self.student.parameters(), lr=self.opt.eta)
        # self.student.load_state_dict(w_init)
        self.student.load_state_dict(torch.load('teacher_w0.pth'))
        # netG.load_state_dict(torch.load('netG.pth'))

        # w_student2 = []
        # for param in self.student.parameters():
        #    w_student2.append(param.data.clone())

        self.experiment = "Trained_Mixup"
        print("Start training {} ...".format(self.experiment))
        logname = os.path.join(self.opt.log_path, 'results' + '_' + self.experiment + '_' + str(self.opt.seed) + '.csv')
        if not os.path.exists(logname):
            with open(logname, 'w') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow(['epoch', 'test acc'])

        student_optim = torch.optim.SGD(self.student.parameters(), lr=self.opt.eta)
        self.student.train()
        self.step = 0
        for epoch in tqdm(range(self.opt.n_epochs)):
            if epoch != 0:
                self.student.train()
                for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                    inputs, targets = inputs.cuda(), targets.long().cuda()
                    # mixed_x, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=1.0)

                    index = torch.randperm(inputs.shape[0]).cuda()
                    lam = netG(inputs, targets.long())
                    lam = torch.unsqueeze(lam, 2)
                    lam = torch.unsqueeze(lam, 3)
                    mixed_x = lam * inputs + (1 - lam) * inputs[index, :]

                    targets_a, targets_b = targets, targets[index]

                    # mixed_x, targets_a, targets_b = mixup_data(inputs, gt_x_2, gt_y_1, gt_y_2, alpha)

                    # first_image = np.array(data.cpu(), dtype='float')
                    # pixels = first_image.reshape((28, 28))
                    # plt.imshow(pixels, cmap='gray')
                    # plt.title("Label {}".format(target.item()))
                    # plt.show()

                    outputs = self.student(mixed_x)
                    loss = mixup_criterion_batch(self.loss_fn, outputs, targets_a, targets_b, lam)
                    # loss = self.loss_fn(outputs, mixed_y.long())

                    student_optim.zero_grad()
                    loss.backward()
                    student_optim.step()

                    self.step += 1
                    if batch_idx % self.opt.log_interval == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, batch_idx * len(inputs), len(self.train_loader.dataset),
                            100. * batch_idx / len(self.train_loader), loss.item()))

                    self.log(mode="train", name="loss", value=loss.item(), step=self.step)

            acc, test_loss = self.test(self.student, test_loader=self.test_loader, epoch=epoch)
            res_student.append(acc)
            res_loss_student.append(test_loss)

            self.adjust_learning_rate(student_optim, epoch)

            with open(logname, 'a') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow([epoch, acc])

            '''
            self.student.eval()
            test = self.student(X_test.cuda()).cpu()

            a, b = plot_classifier(self.student, X.max(axis=0), X.min(axis=0))
            a_student.append(a)
            b_student.append(b)

            if self.opt.data_mode == "mnist" or self.opt.data_mode == "gaussian" or self.opt.data_mode == "moon":
                _, predicted = torch.max(test, dim=1)
                nb_correct = predicted.eq(Y_test.data).cpu().sum().float()
            elif self.opt.data_mode == "cifar10":
                tmp = torch.max(test, dim=1).indices
                nb_correct = torch.where(tmp == Y_test, torch.ones(1), torch.zeros(1)).sum().item()
            else:
                sys.exit()
            acc = nb_correct / X_test.size(0)
            res_student.append(acc)
            '''
            diff = 0
            # diff = torch.linalg.norm(self.teacher.lin.weight - self.student.lin.weight, ord=2) ** 2
            # w_diff_student.append(diff.detach().clone().cpu())

            # w_student = []
            # for param in self.student.parameters():
            #     w_student.append(param.data)
            # diff = weight_diff(w_teacher, w_student)
            # w_diff_student.append(diff)


        sys.exit()


        # mixup baseline
        self.experiment = "Vanilla_Mixup"
        print("Start training {} ...".format(self.experiment))
        logname = os.path.join(self.opt.log_path, 'results' + '_' + self.experiment + '_' + str(self.opt.seed) + '.csv')
        if not os.path.exists(logname):
            with open(logname, 'w') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow(['epoch', 'test acc'])

        res_mixup = []
        res_loss_mixup = []
        a_mixup = []
        b_mixup = []
        loss_mixup = []
        w_diff_mixup = []
        train_loss = 0
        correct = 0
        total = 0
        mixup_baseline_optim = torch.optim.SGD(mixup_baseline.parameters(), lr=self.opt.lr, momentum=0.9, weight_decay=self.opt.decay)
        self.step = 0
        self.best_acc = 0
        for epoch in tqdm(range(self.opt.n_epochs)):
            if epoch != 0:
                mixup_baseline.train()
                for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                    inputs, targets = inputs.cuda(), targets.long().cuda()
                    mixed_x, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=1.0)

                    # first_image = np.array(data.cpu(), dtype='float')
                    # pixels = first_image.reshape((28, 28))
                    # plt.imshow(pixels, cmap='gray')
                    # plt.title("Label {}".format(target.item()))
                    # plt.show()

                    outputs = mixup_baseline(mixed_x)
                    loss = mixup_criterion(self.loss_fn, outputs, targets_a, targets_b, lam)
                    # loss = self.loss_fn(outputs, mixed_y.long())

                    mixup_baseline_optim.zero_grad()
                    loss.backward()
                    mixup_baseline_optim.step()

                    self.step += 1

                    if batch_idx % self.opt.log_interval == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, batch_idx * len(inputs), len(self.train_loader.dataset),
                            100. * batch_idx / len(self.train_loader), loss.item()))
                        self.log(mode="train", name="loss", value=loss.item(), step=self.step)

                '''
                # w_t = self.student.lin.weight

                i = torch.randint(0, nb_batch, size=(1,)).item()
                gt_x_1, gt_y_1 = self.data_sampler(X_train, Y_train, i)

                i = torch.randint(0, nb_batch, size=(1,)).item()
                gt_x_2, gt_y_2 = self.data_sampler(X_train, Y_train, i)

                alpha = np.random.beta(1.0, 1.0)
                # alpha = torch.tensor(alpha, dtype=torch.float).cuda()
                # alpha.requires_grad = True

                mixed_x, targets_a, targets_b = mixup_data(gt_x_1, gt_x_2, gt_y_1, gt_y_2, alpha)
                # mixed_x, targets_a, targets_b = map(Variable, (mixed_x, targets_a, targets_b))

                # inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha, use_cuda)

                mixup_baseline_optim.zero_grad()

                out = mixup_baseline(mixed_x.cuda())
                loss = mixup_criterion(self.loss_fn, out, targets_a.long(), targets_b.long(), alpha)

                # grad = torch.autograd.grad(outputs=loss, inputs=alpha, retain_graph=True)[0]

                loss.backward()
                mixup_baseline_optim.step()

                # mixed_y = gt_y_1 * alpha + gt_y_2 * (1 - alpha)

                # mixup_baseline.update(mixed_x, mixed_y)
                '''

            acc, test_loss = self.test(mixup_baseline, test_loader=self.test_loader, epoch=epoch)
            res_mixup.append(acc)
            res_loss_mixup.append(test_loss)

            self.adjust_learning_rate(mixup_baseline_optim, epoch)

            with open(logname, 'a') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow([epoch, acc])
            '''
            mixup_baseline.eval()
            test = mixup_baseline(X_test.cuda()).cpu()

            a, b = plot_classifier(mixup_baseline, X.max(axis=0), X.min(axis=0))
            a_mixup.append(a)
            b_mixup.append(b)

            if self.opt.data_mode == "mnist" or self.opt.data_mode == "gaussian" or self.opt.data_mode == "moon":
                _, predicted = torch.max(test, dim=1)
                nb_correct = predicted.eq(Y_test.data).cpu().sum().float()
            elif self.opt.data_mode == "cifar10":
                tmp = torch.max(test, dim=1).indices
                nb_correct = torch.where(tmp == Y_test, torch.ones(1), torch.zeros(1)).sum().item()
            else:
                sys.exit()
            acc = nb_correct / X_test.size(0)
            res_mixup.append(acc)
            '''
            # w_mixup = []
            # for param in mixup_baseline.parameters():
            #     w_mixup.append(param.data)
            # diff = weight_diff(w_teacher, w_mixup)
            # w_diff_mixup.append(diff)

            # diff = torch.linalg.norm(self.teacher.lin.weight - mixup_baseline.lin.weight, ord=2) ** 2
            # w_diff_mixup.append(diff.detach().clone().cpu())

            mixup_baseline.train()

        if self.visualize == False:
            fig = plt.figure()
            # plt.plot(w_diff_mixup, c="c", label="Mixup")
            # plt.plot(res_example, c="g", label="SGD")
            plt.plot(res_mixup, c="b", label="Mixup")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.show()

        sys.exit()



         # train example
        self.experiment = "SGD"
        print("Start training {} ...".format(self.experiment))
        logname = os.path.join(self.opt.log_path, 'results' + '_' + self.experiment + '_' + str(self.opt.seed) + '.csv')
        if not os.path.exists(logname):
            with open(logname, 'w') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow(['epoch', 'test acc'])

        res_example = []
        res_loss_example = []
        a_example = []
        b_example = []
        w_diff_example = []
        self.step = 0
        self.best_acc = 0
        example_optim = torch.optim.SGD(example.parameters(), lr=self.opt.lr, momentum=0.9, weight_decay=self.opt.decay)
        for epoch in tqdm(range(self.opt.n_epochs)):
            if epoch != 0:
                self.train(example, self.train_loader, self.loss_fn, example_optim, epoch)

            acc, test_loss = self.test(example, test_loader=self.test_loader, epoch=epoch)
            res_loss_example.append(test_loss)
            res_example.append(acc)

            self.adjust_learning_rate(example_optim, epoch)

            with open(logname, 'a') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow([epoch, acc])

            '''
            example.eval()
            test = example(X_test.cuda()).cpu()

            a, b = plot_classifier(example, X.max(axis=0), X.min(axis=0))
            a_example.append(a)
            b_example.append(b)

            if self.opt.data_mode == "mnist" or self.opt.data_mode == "gaussian" or self.opt.data_mode == "moon":
                _, predicted = torch.max(test, dim=1)
                nb_correct = predicted.eq(Y_test.data).cpu().sum().float()
            elif self.opt.data_mode == "cifar10":
                tmp = torch.max(test, dim=1).indices
                nb_correct = torch.where(tmp == Y_test, torch.ones(1), torch.zeros(1)).sum().item()
            else:
                sys.exit()

            acc = nb_correct / X_test.size(0)
            res_example.append(acc)
            '''

            # w_example = []
            # for param in example.parameters():
            #     w_example.append(param.data)

            # diff = weight_diff(w_teacher, w_example)
            # w_diff_example.append(diff)

            # diff = torch.linalg.norm(w_star - example.lin.weight, ord=2) ** 2
            # w_diff_example.append(diff.detach().clone().cpu())

        if self.visualize == False:
            fig = plt.figure()
            plt.plot(res_loss_example, c="b", label="Teacher (CNN)")
            # plt.plot(w_diff_example, c="b", label="Teacher (CNN)")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.show()

            fig = plt.figure()
            plt.plot(res_example, c="b", label="Teacher (CNN)")
            # plt.plot(w_diff_example, c="b", label="Teacher (CNN)")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.show()


        # train teacher
        accuracies = []
        w_teacher = []
        teacher_optim = torch.optim.SGD(self.teacher.parameters(), lr=self.opt.eta)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(teacher_optim, milestones=[80, 160], gamma=0.1)
        self.opt.n_teacher_runs = 30
        for epoch in tqdm(range(self.opt.n_teacher_runs)):
            self.teacher.train()
            if epoch != 0:
                self.train(self.teacher, train_loader, self.loss_fn, teacher_optim, epoch)

            # acc = self.test(model=self.teacher, test_loader=test_loader)
            acc, test_loss = self.test(model=self.teacher, test_loader=test_loader)
            # accuracies.append(test_loss)
            accuracies.append(acc)

            '''
            self.teacher.eval()
            test = self.teacher(X_test.cuda()).cpu()

            if self.opt.data_mode == "mnist" or self.opt.data_mode == "gaussian" or self.opt.data_mode == "moon" or self.opt.data_mode == "linearly_seperable":
                _, predicted = torch.max(test, dim=1)
                nb_correct = predicted.eq(Y_test.data).cpu().sum().float()

                # tmp = torch.where(test > 0.5, torch.ones(1), torch.zeros(1))
                # nb_correct = torch.where(tmp.view(-1) == Y_test, torch.ones(1), torch.zeros(1)).sum().item()
            elif self.opt.data_mode == "cifar10":
                tmp = torch.max(test, dim=1).indices
                nb_correct = torch.where(tmp == Y_test, torch.ones(1), torch.zeros(1)).sum().item()
            else:
                sys.exit()
            acc = nb_correct / X_test.size(0)

            accuracies.append(acc)
            print("Accuracy:", acc.item())
            self.scheduler.step()

            self.teacher.train()
            '''

        if self.visualize == False:
            fig = plt.figure()
            plt.plot(accuracies, c="b", label="Teacher (CNN)")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.show()
            # plt.close()

        '''
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
        '''

        if self.visualize == True:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(14, 5.8)
            # a, b = plot_classifier(self.teacher, X.max(axis=0), X.min(axis=0))
            ax1.plot(res_student, 'r', label="Mixup Trained")
            ax1.plot(res_mixup, 'b', label="Mixup")
            ax1.plot(res_example, 'g', label="SGD")
            # ax1.plot(w_diff_example, 'go', label="sgd linear classifier", alpha=0.5)
            # ax1.plot(w_diff_baseline, 'bo', label="%s & baseline" % self.opt.teaching_mode, alpha=0.5)
            # ax1.plot(w_diff_student, 'ro', label="%s & linear classifier" % self.opt.teaching_mode, alpha=0.5)
            # ax1.plot(w_diff_mixup, 'co', label="%s & mixup classifier" % self.opt.teaching_mode, alpha=0.5)
            # ax1.axhline(y=teacher_acc, color='k', linestyle='-', label="teacher accuracy")
            ax1.legend(loc="upper left")
            ax1.set_title(str(self.opt.data_mode) + "Test Accuracy (class : " + str(self.opt.class_1) + ", " + str(self.opt.class_2) + ")")
            #ax1.set_aspect('equal')
            # ax1.close()

            ax2.plot(res_loss_student, c='r', label="Mixup Trained")
            ax2.plot(res_loss_mixup, c='b', label="Mixup")
            ax2.plot(res_loss_example, c='g', label="SGD")
            ax2.set_title(str(self.opt.data_mode) + "Test Loss (class : " + str(self.opt.class_1) + ", " + str(self.opt.class_2) + ")")
            # ax2.xlabel("Iteration")
            # ax2.ylabel("Loss")
            ax2.legend(loc="upper right")

            save_folder = os.path.join(self.opt.log_path, "imgs")
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            img_path = os.path.join(save_folder, "results_mnist_blackbox_mixup_cnn.png")

            plt.savefig(img_path)
            plt.show()
            # plt.close()

            '''
            os.chdir(CONF.PATH.OUTPUT)
            subprocess.call([
                'ffmpeg', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
                'video_name.mp4'
            ])
            for file_name in glob.glob("*.png"):
                os.remove(file_name)
            '''

        sys.exit()

        # train student
        if self.opt.data_mode == "moon":
            netG = blackbox_mixup.Generator_moon(self.opt, self.teacher, tmp_student).cuda()
        else:
            netG = blackbox_mixup.Generator(self.opt).cuda()

        netG.apply(weights_init)

        optimizer_G = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999), eps=1e-08, amsgrad=False)
        # optimizer_G = torch.optim.SGD(netG.parameters(), lr=self.opt.eta)
        student_optim = torch.optim.SGD(self.student.parameters(), lr=self.opt.eta)
        unrolled_optimizer = blackbox_mixup.UnrolledBlackBoxOptimizer(opt=self.opt, teacher=self.teacher, student=tmp_student, generator=netG, X=X_train.cuda(), y=Y_train.cuda(), proj_matrix=None)
        adversarial_loss = torch.nn.MSELoss()
        res_student = []
        res_loss_student = []
        a_student = []
        b_student = []
        loss_student = []
        cls_loss = []
        loss_g = []
        loss_d = []
        w_diff_student = []
        # w, h = generator.linear.weight.shape

        generated_samples = np.zeros(2)

        self.step = 0

        w_init = self.student.state_dict()
        w_student = []
        for param in self.student.parameters():
            w_student.append(param.data.clone())
        # new_weight = w_init

        self.opt.n_unroll = 6000
        for idx in tqdm(range(self.opt.n_unroll)):
            if idx != 0:

                i = torch.randint(0, nb_batch, size=(1,)).item()
                gt_x_1, gt_y_1 = self.data_sampler(X_train, Y_train, i)

                i = torch.randint(0, nb_batch, size=(1,)).item()
                gt_x_2, gt_y_2 = self.data_sampler(X_train, Y_train, i)

                x = torch.cat((gt_x_1, gt_x_2), dim=1)
                alpha = netG(x, gt_y_1.long(), gt_y_2.long())

                mixed_x = alpha * gt_x_1 + (1 - alpha) * gt_x_2
                # mixed_y = alpha * gt_y_1 + (1 - alpha) * gt_y_2

                optimizer_G.zero_grad()
                out = self.student(mixed_x)
                # loss = mixup_criterion(self.loss_fn, out, targets_a.float(), targets_b.float(), alpha)
                loss = alpha * self.loss_fn(out, gt_y_1.long()) + (1 - alpha) * self.loss_fn(out, gt_y_2.long())
                loss = loss.to(torch.float32)

                # grad = torch.autograd.grad(loss, alpha, create_graph=True, retain_graph=True)
                # grad = torch.autograd.grad(alpha, model_paramters, create_graph=True, retain_graph=True)
                # grad = torch.autograd.grad(loss, model_paramters, create_graph=True, retain_graph=True)

                loss.backward() # create_graph=True, retain_graph=True)
                # loss.backward()
                optimizer_G.step()

                # i = torch.randint(0, nb_batch, size=(1,)).item()
                # gt_x_1, gt_y_1 = self.data_sampler(X_train, Y_train, i)

                # i = torch.randint(0, nb_batch, size=(1,)).item()
                # gt_x_2, gt_y_2 = self.data_sampler(X_train, Y_train, i)

                # x = torch.cat((gt_x_1, gt_x_2), dim=1)
                alpha = netG(x, gt_y_1.long(), gt_y_2.long())

                mixed_x = alpha * gt_x_1 + (1 - alpha) * gt_x_2

                student_optim.zero_grad()

                out = self.student(mixed_x)
                # loss = mixup_criterion(self.loss_fn, out, targets_a.float(), targets_b.float(), alpha)
                loss = alpha * self.loss_fn(out, gt_y_1.long()) + (1 - alpha) * self.loss_fn(out, gt_y_2.long())
                loss = loss.to(torch.float32)
                loss.backward()

                student_optim.step()

                '''
                w_t = netG.state_dict()
                gradients, loss, loss_tmp = unrolled_optimizer(w_t, w_init=w_init)

                # cls_loss = cls_loss + loss_tmp
                cls_loss = loss_tmp

                loss_student.append(loss.item())

                with torch.no_grad():
                    for p, g in zip(netG.parameters(), gradients):
                        p.grad = g

                optimizer_G.step()
                '''
        '''
        for idx in tqdm(range(self.opt.n_unroll)):
            if idx != 0:

                w_t = netG.state_dict()
                gradients, loss, loss_tmp = unrolled_optimizer(w_t, w_init=w_init)

                # cls_loss = cls_loss + loss_tmp
                cls_loss = loss_tmp

                loss_student.append(loss.item())

                with torch.no_grad():
                    for p, g in zip(netG.parameters(), gradients):
                        p.grad = g

                optimizer_G.step()
        '''

        # student_optim = torch.optim.SGD(self.student.parameters(), lr=self.opt.eta)
        # self.student.load_state_dict(w_init)
        self.student.load_state_dict(torch.load('teacher_w0.pth'))

        self.student.train()
        train_loss = []
        for idx in tqdm(range(self.opt.n_iter)):
            if idx != 0:
                # w_t = self.student.lin.weight

                '''
                y = torch.randint(0, 2, (1,), dtype=torch.float).cuda()
                b = Y_train.cuda() == y
                indices = b.nonzero()
                idx = torch.randint(0, len(indices), (1,))
                gt_x = X_train[indices[idx].squeeze(0)].cuda()
                '''
                i = torch.randint(0, nb_batch, size=(1,)).item()
                i_min = i * self.opt.batch_size
                i_max = (i + 1) * self.opt.batch_size

                gt_x_1 = X_train[i_min:i_max].cuda()
                gt_y_1 = Y_train[i_min:i_max].cuda()

                i = torch.randint(0, nb_batch, size=(1,)).item()
                i_min = i * self.opt.batch_size
                i_max = (i + 1) * self.opt.batch_size

                gt_x_2 = X_train[i_min:i_max].cuda()
                gt_y_2 = Y_train[i_min:i_max].cuda()

                # z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, gt_x.shape)))
                # z = Variable(torch.randn(gt_x.shape)).cuda()
                # z = Variable(torch.randn((self.opt.batch_size, self.opt.latent_dim))).cuda()

                # x = torch.cat((w_t, w_t-w_star, gt_x), dim=1)
                x = torch.cat((gt_x_1, gt_x_2), dim=1)
                alpha = netG(x, gt_y_1.long(), gt_y_2.long())

                print(alpha)
                # alpha = np.random.beta(1.0, 1.0)

                mixed_x, targets_a, targets_b = mixup_data(gt_x_1, gt_x_2, gt_y_1, gt_y_2, alpha)
                # mixed_x, targets_a, targets_b = map(Variable, (mixed_x, targets_a, targets_b))

                # self.student.train()

                # loss = mixup_criterion(self.loss_fn, out, targets_a.float(), targets_b.float(), alpha)
                # loss = self.loss_fn(out, generated_y.float())

                # grad = torch.autograd.grad(loss, self.student.lin.weight, create_graph=True)
                # new_weight = self.student.lin.weight - 0.001 * grad[0]
                # new_weight = new_weight - 0.001 * grad[0]
                # self.student.lin.weight = torch.nn.Parameter(new_weight.cuda())

                student_optim.zero_grad()
                out = self.student(mixed_x)
                loss = mixup_criterion(self.loss_fn, out, targets_a.long(), targets_b.long(), alpha)

                # grad = torch.autograd.grad(outputs=loss, inputs=alpha, retain_graph=True)[0]

                loss.backward()
                student_optim.step()

                # mixed_y = targets_a * alpha + targets_b * (1 - alpha)

                # self.student.update(mixed_x, mixed_y)
                train_loss.append(loss.item())

            acc, test_loss = self.test(self.student, test_loader=test_loader)
            res_student.append(acc)
            res_loss_student.append(test_loss)

            '''
            self.student.eval()
            test = self.student(X_test.cuda()).cpu()

            a, b = plot_classifier(self.student, X.max(axis=0), X.min(axis=0))
            a_student.append(a)
            b_student.append(b)

            if self.opt.data_mode == "mnist" or self.opt.data_mode == "gaussian" or self.opt.data_mode == "moon":
                _, predicted = torch.max(test, dim=1)
                nb_correct = predicted.eq(Y_test.data).cpu().sum().float()
            elif self.opt.data_mode == "cifar10":
                tmp = torch.max(test, dim=1).indices
                nb_correct = torch.where(tmp == Y_test, torch.ones(1), torch.zeros(1)).sum().item()
            else:
                sys.exit()
            acc = nb_correct / X_test.size(0)
            res_student.append(acc)
            '''
            diff = 0
            # diff = torch.linalg.norm(self.teacher.lin.weight - self.student.lin.weight, ord=2) ** 2
            # w_diff_student.append(diff.detach().clone().cpu())

            # w_student = []
            # for param in self.student.parameters():
            #     w_student.append(param.data)
            # diff = weight_diff(w_teacher, w_student)
            # w_diff_student.append(diff)

            self.student.train()

        if self.visualize == True:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(14, 5.8)
            # a, b = plot_classifier(self.teacher, X.max(axis=0), X.min(axis=0))
            ax1.plot(res_student, 'r', label="Mixup Trained")
            ax1.plot(res_mixup, 'b', label="Mixup")
            ax1.plot(res_example, 'g', label="SGD")
            # ax1.plot(w_diff_example, 'go', label="sgd linear classifier", alpha=0.5)
            # ax1.plot(w_diff_baseline, 'bo', label="%s & baseline" % self.opt.teaching_mode, alpha=0.5)
            # ax1.plot(w_diff_student, 'ro', label="%s & linear classifier" % self.opt.teaching_mode, alpha=0.5)
            # ax1.plot(w_diff_mixup, 'co', label="%s & mixup classifier" % self.opt.teaching_mode, alpha=0.5)
            # ax1.axhline(y=teacher_acc, color='k', linestyle='-', label="teacher accuracy")
            ax1.legend(loc="upper left")
            ax1.set_title(str(self.opt.data_mode) + "Test Accuracy (class : " + str(self.opt.class_1) + ", " + str(self.opt.class_2) + ")")
            #ax1.set_aspect('equal')
            # ax1.close()

            ax2.plot(res_loss_student, c='r', label="Mixup Trained")
            ax2.plot(res_loss_mixup, c='b', label="Mixup")
            ax2.plot(res_loss_example, c='g', label="SGD")
            ax2.set_title(str(self.opt.data_mode) + "Test Loss (class : " + str(self.opt.class_1) + ", " + str(self.opt.class_2) + ")")
            # ax2.xlabel("Iteration")
            # ax2.ylabel("Loss")
            ax2.legend(loc="upper right")

            save_folder = os.path.join(self.opt.log_path, "imgs")
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            img_path = os.path.join(save_folder, "results_mnist_blackbox_mixup_cnn.png")

            plt.savefig(img_path)
            plt.show()
            # plt.close()

            '''
            os.chdir(CONF.PATH.OUTPUT)
            subprocess.call([
                'ffmpeg', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
                'video_name.mp4'
            ])
            for file_name in glob.glob("*.png"):
                os.remove(file_name)
            '''

    def data_sampler(self, X, y, i):
        i_min = i * self.opt.batch_size
        i_max = (i + 1) * self.opt.batch_size

        x = X[i_min:i_max].cuda()
        y = y[i_min:i_max].cuda()

        return x, y

    def train(self, model, train_loader, loss_fn, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):

            # first_image = np.array(data.cpu(), dtype='float')
            # pixels = first_image.reshape((28, 28))
            # plt.imshow(pixels, cmap='gray')
            # plt.title("Label {}".format(target.item()))
            # plt.show()

            data, target = data.cuda(), target.long().cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            self.step += 1
            if batch_idx % self.opt.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                self.log(mode="train", name="loss", value=loss.item(), step=self.step)

    def val(self, model, train_loader, loss_fn, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):

            # first_image = np.array(data.cpu(), dtype='float')
            # pixels = first_image.reshape((28, 28))
            # plt.imshow(pixels, cmap='gray')
            # plt.title("Label {}".format(target.item()))
            # plt.show()

            data, target = data.cuda(), target.long().cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % self.opt.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                self.log(mode="train", name="loss", value=loss.item())

    def test(self, model, test_loader, epoch):
        model.eval()
        test_loss = 0
        correct = 0
        loss_fn = nn.CrossEntropyLoss(reduction='sum')
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.cuda(), target.cuda()
                output = model(data)

                test_loss += loss_fn(output, target.long()).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        acc = correct / len(test_loader.dataset)
        self.log(mode="test", name="acc", value=acc, step=epoch)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        self.log(mode="test", name="loss", value=test_loss, step=epoch)

        if epoch == 0 or acc > self.best_acc:
            self.save_model(model=model, name=self.experiment)
        if acc > self.best_acc:
            best_acc = acc

        return acc, test_loss

    def make_results_img_2d(self, X, Y, a_student, b_student, generated_samples, generated_labels, w_diff_example, w_diff_baseline, w_diff_student, loss_student, loss_g, loss_d, epoch=None):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.set_size_inches(20, 5.8)
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

        ax3.plot(loss_g, c='b', label="netG loss")
        ax3.plot(loss_d, c='g', label="netD loss")
        ax3.plot(loss_student, c='r', label="generator loss")
        ax3.set_title(str(self.opt.data_mode) + "Model (class : " + str(self.opt.class_1) + ", " + str(self.opt.class_2) + ")")
        # ax3.xlabel("Iteration")
        # ax3.ylabel("Loss")
        ax3.legend(loc="upper right")

        fig.suptitle('Epoch {}'.format(epoch), fontsize=16)

        save_folder = os.path.join(self.opt.log_path, "imgs")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        img_path = os.path.join(save_folder, "results_{}.png".format(epoch))

        plt.savefig(img_path)
        # plt.show()
        plt.close()

    def make_results_img(self, X, Y, a_student, b_student, generated_samples, generated_labels, w_diff_example, w_diff_baseline, w_diff_student, loss_student, loss_g, loss_d, epoch, proj_matrix):
        # unproj_matrix = np.linalg.pinv(proj_matrix)
        n_rows = 10
        indices = torch.randint(0, len(generated_samples), (n_rows**2,))
        labels = generated_labels[indices]
        samples = generated_samples[indices]

        # gen_imgs = samples @ unproj_matrix

        img_shape = (1, 28, 28)
        gen_imgs = samples
        im = np.reshape(samples, (samples.shape[0], *img_shape))
        im = torch.from_numpy(im)

        save_folder = os.path.join(self.opt.log_path, "imgs")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        grid = make_grid(im, nrow=10, normalize=True)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(grid.permute(1, 2, 0).data, cmap='binary')
        ax.axis('off')
        ax.set_title("Fake Images, Label", )
        img_path = os.path.join(save_folder, "results_{}_imgs.png".format(epoch))
        plt.savefig(img_path)
        plt.close()

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(13, 5.8)
        # ax1.plot(res_example, 'go', label="linear classifier", alpha=0.5)
        # ax1.plot(res_baseline[:i+1], 'bo', label="%s & baseline" % self.opt.teaching_mode, alpha=0.5)
        # ax1.plot(res_student[:i+1], 'ro', label="%s & linear classifier" % self.opt.teaching_mode, alpha=0.5)
        ax1.plot(w_diff_example, 'go', label="linear classifier", alpha=0.5)
        ax1.plot(w_diff_baseline, 'bo', label="%s & baseline" % self.opt.teaching_mode, alpha=0.5)
        ax1.plot(w_diff_student, 'ro', label="%s & linear classifier" % self.opt.teaching_mode, alpha=0.5)
        # plt.axhline(y=teacher_acc, color='k', linestyle='-', label="teacher accuracy")
        ax1.legend(loc="upper right")
        ax1.set_title("Test Set Accuracy")
        # ax1.set_aspect('equal')

        ax2.plot(loss_g, c='b', label="netG loss")
        ax2.plot(loss_d, c='g', label="netD loss")
        ax2.plot(loss_student, c='r', label="generator loss")
        ax2.set_title(str(self.opt.data_mode) + "Model (class : " + str(self.opt.class_1) + ", " + str(self.opt.class_2) + ")")
        # ax2.xlabel("Iteration")
        # ax2.ylabel("Loss")
        ax2.legend(loc="upper right")

        img_path = os.path.join(save_folder, "results_{}_w_diff.png".format(epoch))
        plt.savefig(img_path)
        plt.close()

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

    def log(self, mode, name, value, step):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        writer.add_scalar("{}/{}/{}".format(self.experiment, mode, name), value, step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.opt.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self, model, name):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.opt.log_path, "weights")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        save_path = os.path.join(save_folder, "best_model_{}.pth".format(name))
        to_save = model.state_dict()
        torch.save(to_save, save_path)

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
