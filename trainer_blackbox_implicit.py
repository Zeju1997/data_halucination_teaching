from __future__ import absolute_import, division, print_function

import numpy as np
import time
import json

import copy
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

import itertools

from torch.optim.lr_scheduler import StepLR

from datasets import MoonDataset

from datasets import BaseDataset

import networks.cgan_cifar100 as cgan
import networks.unrolled_optimizer as unrolled
import networks.blackbox_mixup_cnn as blackbox_mixup
import networks.blackbox_implicit as blackbox_implicit
# import networks

from utils import HSIC

from sklearn.datasets import make_moons, make_classification
from sklearn.model_selection import train_test_split

import subprocess
import glob

from utils.utils import progress_bar

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
    loss = torch.mean(loss)
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

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

activation = {}


class EstimatorCV():
    def __init__(self, feature_num, class_num):
        super(EstimatorCV, self).__init__()
        self.class_num = class_num
        self.CoVariance = torch.zeros(class_num, feature_num, feature_num).cuda()
        self.Ave = torch.zeros(class_num, feature_num).cuda()
        self.Amount = torch.zeros(class_num).cuda()

    def update_CV(self, features, labels):
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        NxCxFeatures = features.view(
            N, 1, A
        ).expand(
            N, C, A
        )
        onehot = torch.zeros(N, C).cuda()
        onehot.scatter_(1, labels.view(-1, 1), 1)

        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        var_temp = features_by_sort - \
                   ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

        var_temp = torch.bmm(
            var_temp.permute(1, 2, 0),
            var_temp.permute(1, 0, 2)
        ).div(Amount_CxA.view(C, A, 1).expand(C, A, A))

        sum_weight_CV = onehot.sum(0).view(C, 1, 1).expand(C, A, A)

        sum_weight_AV = onehot.sum(0).view(C, 1).expand(C, A)

        weight_CV = sum_weight_CV.div(
            sum_weight_CV + self.Amount.view(C, 1, 1).expand(C, A, A)
        )
        weight_CV[weight_CV != weight_CV] = 0

        weight_AV = sum_weight_AV.div(
            sum_weight_AV + self.Amount.view(C, 1).expand(C, A)
        )
        weight_AV[weight_AV != weight_AV] = 0

        additional_CV = weight_CV.mul(1 - weight_CV).mul(
            torch.bmm(
                (self.Ave - ave_CxA).view(C, A, 1),
                (self.Ave - ave_CxA).view(C, 1, A)
            )
        )

        self.CoVariance = (self.CoVariance.mul(1 - weight_CV) + var_temp
                      .mul(weight_CV)).detach() + additional_CV.detach()

        self.Ave = (self.Ave.mul(1 - weight_AV) + ave_CxA.mul(weight_AV)).detach()

        self.Amount += onehot.sum(0)


class Trainer:
    def __init__(self, options):
        self.opt = options

        self.opt.model_name = "blackbox_implicit_" + self.opt.data_mode + "_" + str(self.opt.n_weight_update) + '_' + str(self.opt.n_z_update) + '_' + str(self.opt.model)

        self.opt.log_path = os.path.join(CONF.PATH.LOG, self.opt.model_name)
        if not os.path.exists(self.opt.log_path):
            os.makedirs(self.opt.log_path)

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
                                         (0.2470, 0.2435, 0.2616)),
                ])
            else:
                transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2470, 0.2435, 0.2616)),
                ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2470, 0.2435, 0.2616)),
            ])
            dataset = torchvision.datasets.CIFAR10(root=CONF.PATH.DATA, train=True, download=True, transform=transform_train)
            self.test_dataset = torchvision.datasets.CIFAR10(root=CONF.PATH.DATA, train=False, download=True, transform=transform_test)

            self.train_dataset, self.val_dataset = random_split(dataset, [40000, 10000])

            self.train_loader = DataLoader(self.train_dataset, batch_size=self.opt.batch_size, num_workers=self.opt.num_workers, pin_memory=True, shuffle=True, drop_last=True)
            self.val_loader = DataLoader(self.val_dataset, batch_size=self.opt.batch_size, num_workers=self.opt.num_workers, pin_memory=True, shuffle=True, drop_last=True)
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.opt.batch_size, num_workers=self.opt.num_workers, pin_memory=True, shuffle=False, drop_last=True)

            self.loader = DataLoader(dataset, batch_size=self.opt.batch_size, num_workers=self.opt.num_workers, pin_memory=True, shuffle=True, drop_last=True)

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
            dataset = torchvision.datasets.CIFAR100(root=CONF.PATH.DATA, train=True, download=True, transform=transform_train)
            self.test_dataset = torchvision.datasets.CIFAR100(root=CONF.PATH.DATA, train=False, download=True, transform=transform_test)

            self.train_dataset, self.val_dataset = random_split(dataset, [40000, 10000])

            self.train_loader = DataLoader(self.train_dataset, batch_size=self.opt.batch_size, num_workers=self.opt.num_workers, pin_memory=True, shuffle=True)
            self.val_loader = DataLoader(self.val_dataset, batch_size=self.opt.batch_size, num_workers=self.opt.num_workers, pin_memory=True, shuffle=True)
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.opt.batch_size, num_workers=self.opt.num_workers, pin_memory=True, shuffle=False)

            self.loader = DataLoader(dataset, batch_size=self.opt.batch_size, num_workers=self.opt.num_workers, pin_memory=True, shuffle=True)

        elif self.opt.data_mode == "mnist":
            # MNIST normalizing
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)), # transforms.Normalize((0.1307,), (0.3081,)),
            ])
            self.train_dataset = torchvision.datasets.MNIST(root=CONF.PATH.DATA, train=True, download=True, transform=transform)
            # train, valid = random_split(train_dataset, [50000, 10000])
            self.test_dataset = torchvision.datasets.MNIST(root=CONF.PATH.DATA, train=False, download=True, transform=transform)
            self.loader = DataLoader(self.train_dataset, batch_size=self.opt.batch_size, shuffle=True, num_workers=self.opt.num_workers, pin_memory=True)
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.opt.batch_size, shuffle=False, num_workers=self.opt.num_workers, pin_memory=True)

        else:
            print("Unrecognized data!")
            sys.exit()

        self.get_teacher_student()

        self.writers = {}
        for mode in ["train", "val", "test"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.opt.log_path, mode))

        # self.loss_fn = nn.MSELoss()
        self.loss_fn = nn.CrossEntropyLoss()

        self.step = 0
        self.best_acc = 0
        self.best_test_loss = 0
        self.init_train_loss = 0
        self.init_test_loss = 0
        self.init_feat_sim = 0

        self.opt.experiment = "teacher"

        self.estimator = EstimatorCV(feature_num=512, class_num=self.opt.n_classes)

        # self.query_set_1, self.query_set_2 = self.get_query_set()
        # self.query_set = self.get_query_set()

    def get_teacher_student(self):
        self.teacher = networks.Net().cuda()
        # self.teacher = networks.MLP(n_in=self.opt.n_in, num_classes=self.opt.n_classes).cuda()
        self.teacher.apply(initialize_weights)
        self.teacher_fc = networks.FullLayer(feature_dim=self.teacher.feature_num, n_classes=self.opt.n_classes).cuda()
        torch.save(self.teacher.state_dict(), os.path.join(self.opt.log_path, 'teacher_w0.pth'))
        self.teacher_fc.apply(initialize_weights)
        torch.save(self.teacher_fc.state_dict(), os.path.join(self.opt.log_path, 'teacher_fc_w0.pth'))

        # self.teacher.load_state_dict(torch.load('teacher.pth'))
        # path = os.path.join(self.opt.log_path, 'weights/best_model_SGD.pth')

        self.student = networks.networks.Net().cuda()
        # self.student = networks.MLP(n_in=self.opt.n_in, num_classes=self.opt.n_classes).cuda()
        self.student_fc = networks.FullLayer(feature_dim=self.student.feature_num, n_classes=self.opt.n_classes).cuda()
        # self.student.load_state_dict(torch.load(path))
        # self.student.model.avgpool.register_forward_hook(self.get_activation('latent'))
        self.baseline = networks.Net().cuda()
        # self.baseline = networks.MLP(n_in=self.opt.n_in, num_classes=self.opt.n_classes).cuda()
        self.baseline_fc = networks.FullLayer(feature_dim=self.baseline.feature_num, n_classes=self.opt.n_classes).cuda()

        # load teacher weights
        self.baseline.load_state_dict(self.teacher.state_dict())
        self.baseline_fc.load_state_dict(self.teacher_fc.state_dict())

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

    def adjust_learning_rate(self, optimizer, iter):
        """decrease the learning rate at 100 and 150 epoch"""
        lr = self.opt.lr
        if iter == 20000 or iter == 30000 or iter == 37500: # 100
            lr /= 10
            self.opt.lr = lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def get_query_set(self):
        """decrease the learning rate at 100 and 150 epoch"""
        query_set_1 = torch.empty(self.opt.n_query_classes, self.opt.channels, self.opt.img_size, self.opt.img_size)
        query_set_2 = torch.empty(self.opt.n_query_classes, self.opt.channels, self.opt.img_size, self.opt.img_size)
        val_iter = iter(self.val_loader)
        for i in range(self.opt.n_classes):
            while True:
                try:
                    (inputs, targets) = val_iter.next()
                except:
                    val_iter = iter(self.val_loader)
                    (inputs, targets) = val_iter.next()
                idx = ((targets == i).nonzero(as_tuple=True)[0])
                if idx.nelement() == 0:
                    pass
                else:
                    idx = idx[0]
                    query_set_1[i, :] = inputs[idx, :]
                    break

        for i in range(self.opt.n_classes):
            while True:
                try:
                    (inputs, targets) = val_iter.next()
                except:
                    val_iter = iter(self.val_loader)
                    (inputs, targets) = val_iter.next()
                idx = ((targets == i).nonzero(as_tuple=True)[0])
                if idx.nelement() == 0:
                    pass
                else:
                    idx = idx[0]
                    query_set_2[i, :] = inputs[idx, :]
                    break

        return query_set_1.cuda(), query_set_2.cuda()

    def get_query_set2(self):
        """decrease the learning rate at 100 and 150 epoch"""
        n_samples = 10
        query_set = torch.zeros(self.opt.n_query_classes, n_samples, self.opt.channels, self.opt.img_size, self.opt.imgF_size)

        sample_classes = np.random.choice(self.opt.n_classes, self.opt.n_query_classes)

        val_iter = iter(self.val_loader)
        count = 0
        for i, label in enumerate(sample_classes):
            while count < n_samples:
                try:
                    (inputs, targets) = val_iter.next()
                except:
                    val_iter = iter(self.val_loader)
                    (inputs, targets) = val_iter.next()
                idx = ((targets == label).nonzero(as_tuple=True)[0])
                if idx.nelement() == 0:
                    pass
                else:
                    idx = idx[0]
                    query_set[i, count, :] = inputs[idx, :]
                    count = count + 1
                    pass
            count = 0

        return query_set.cuda()

    def main(self):
        """Run a single epoch of training and validation
        """
        '''
        from time import time
        import multiprocessing as mp

        for num_workers in range(0, mp.cpu_count(), 2):
            train_loader = DataLoader(self.train_dataset, shuffle=True, num_workers=num_workers, batch_size=self.opt.batch_size, pin_memory=True)
            start = time()
            for epoch in range(1, 3):
                for i, data in enumerate(train_loader, 0):
                    pass
            end = time()
            print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
        
        import torch
        from torchvision import transforms
        import torchvision.datasets as datasets
        import matplotlib.pyplot as plt
        from sklearn.cluster import KMeans


        kmeans = KMeans(n_clusters=10)

        X = self.val_dataset.dataset.data

        # KMmodel = kmeans.fit(self.val_dataset.data.numpy())
        KMmodel = kmeans.fit(X)
        print("cluster centers", kmeans.labels_)
        print("cluster centers", kmeans.cluster_centers_)
        sys.exit()
        
        '''

        print(self.opt)

        print("Training")
        # self.set_train()

        example = networks.Net().cuda()
        # example = networks.MLP(n_in=self.opt.n_in, num_classes=self.opt.n_classes).cuda()
        example_fc = networks.FullLayer(feature_dim=example.feature_num, n_classes=self.opt.n_classes).cuda()

        if self.opt.train_sgd == True:
            # train example
            self.opt.experiment = "SGD"
            print("Start training {} ...".format(self.opt.experiment))
            logname = os.path.join(self.opt.log_path, 'results' + '_' + self.opt.experiment + '_' + self.opt.model + '_' + str(self.opt.seed) + '.csv')
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
            # example_optim = torch.optim.SGD(example.parameters(), lr=self.opt.lr, momentum=0.9, weight_decay=self.opt.decay)

            pdist = torch.nn.PairwiseDistance(p=2)

            example.load_state_dict(torch.load(os.path.join(self.opt.log_path, 'teacher_w0.pth')))
            example_fc.load_state_dict(torch.load(os.path.join(self.opt.log_path, 'teacher_fc_w0.pth')))
            example_optim = torch.optim.SGD([{'params': example.parameters()}, {'params': example_fc.parameters()}], lr=0.001)
            # example_optim = torch.optim.SGD([{'params': example.parameters()}, {'params': example_fc.parameters()}],
            #                                 lr=self.opt.lr,
            #                                 momentum=self.opt.momentum, nesterov=self.opt.nesterov,
            #                                 weight_decay=self.opt.weight_decay)

            # cosine learning rate
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(example_optim, len(self.train_loader)*self.opt.n_epochs)

            step = 0
            for epoch in tqdm(range(self.opt.n_epochs)):
                if epoch != 0:
                    # self.train(example, example_fc, self.loader, self.loss_fn, example_optim, epoch)
                    train_loss = 0
                    correct = 0
                    total = 0
                    example.train()
                    example_fc.train()
                    for batch_idx, (data, target) in enumerate(self.loader):
                        data, target = data.cuda(), target.long().cuda()

                        example_optim.zero_grad()
                        z = example(data)
                        # z1 = z
                        output = example_fc(z)
                        loss = self.loss_fn(output, target)
                        loss.backward()
                        example_optim.step()
                        # z2 = example(data)
                        # dist = pdist(z1, z2).max()
                        # print("dist", dist)

                        step = step + 1
                        self.adjust_learning_rate(example_optim, step)

                        train_loss += loss.item()
                        _, predicted = torch.max(output.data, 1)
                        total += target.size(0)
                        correct += predicted.eq(target.data).cpu().sum()

                        progress_bar(batch_idx, len(self.loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

                    print('Epoch: %d | Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)' % (epoch, train_loss/(batch_idx+1), 100.*correct/total, correct, total))

                acc, test_loss = self.test(example, example_fc, test_loader=self.test_loader, epoch=epoch)
                res_loss_example.append(test_loss)
                res_example.append(acc)

                # self.adjust_learning_rate(example_optim, epoch)

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

            if self.visualize == False:
                fig, (ax1, ax2) = plt.subplots(1, 2)
                fig.set_size_inches(12, 5.8)
                ax1.plot(res_loss_example, c="b", label="Teacher (CNN)")
                ax1.xlabel("Epoch")
                ax1.ylabel("Loss")
                ax1.legend()

                ax2.plot(res_example, c="b", label="Teacher (CNN)")
                ax2.xlabel("Epoch")
                ax2.ylabel("Accuracy")
                ax2.legend()
                ax2.show()

        sys.exit()

        if self.opt.train_student == True:
            # student
            self.opt.experiment = "Student"
            print("Start training {} ...".format(self.opt.experiment))
            logname = os.path.join(self.opt.log_path, 'results' + '_' + self.opt.experiment + '_' + self.opt.model + '_' + str(self.opt.seed) + '_' + self.opt.data_mode + "_" + str(self.opt.n_weight_update) + '_' + str(self.opt.n_z_update) + '_' + str(self.opt.epsilon) + '.csv')
            if not os.path.exists(logname):
                with open(logname, 'w') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    logwriter.writerow(['epoch', 'test acc'])

            res_mixup = []
            res_loss_mixup = []

            tmp_student = networks.FullLayer(feature_dim=self.teacher.feature_num, n_classes=self.opt.n_classes).cuda()

            unrolled_optimizer = blackbox_implicit.UnrolledBlackBoxOptimizer(opt=self.opt, loader=self.loader, fc=tmp_student)

            res_student = []
            res_loss_student = []
            cls_loss = []

            w_init = self.student.state_dict()
            # new_weight = w_init
            train_loss = []

            self.step = 0
            self.best_acc = 0
            self.best_test_loss = 0
            self.init_train_loss = 0
            self.init_test_loss = 0
            self.init_feat_sim = 0

            # loader_eval = iter(self.val_loader)
            # avg_train_loss = 0
            # tmp_train_loss = 0
            # feat_sim = 0

            self.student.load_state_dict(torch.load(os.path.join(self.opt.log_path, 'teacher_w0.pth')))
            self.student_fc.load_state_dict(torch.load(os.path.join(self.opt.log_path, 'teacher_fc_w0.pth')))
            student_optim = torch.optim.SGD([{'params': self.student.parameters()}, {'params': self.student_fc.parameters()}], lr=0.001)
            # student_optim = torch.optim.SGD([{'params': self.student.parameters()}, {'params': self.student_fc.parameters()}],
            #                                 lr=self.opt.lr,
            #                                 momentum=self.opt.momentum, nesterov=self.opt.nesterov,
            #                                 weight_decay=self.opt.weight_decay)
            # student_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(student_optim, len(self.train_loader)*self.opt.n_epochs)

            # student_parameters = list(self.student.parameters()) + list(self.student_fc.parameters())
            # train_loss = []
            # train_loss2 = []
            for epoch in range(self.opt.n_epochs):
                if epoch != 0:
                    self.student.train()
                    self.student_fc.train()
                    batch_idx = 0

                    train_loss = 0
                    correct = 0
                    total = 0

                    for (inputs, targets) in self.loader:

                        inputs, targets = inputs.cuda(), targets.long().cuda()
                        student_optim.zero_grad()

                        # model_mdl = copy.deepcopy(self.student)
                        z = self.student(inputs)

                        torch.save(self.student_fc.state_dict(), os.path.join(self.opt.log_path, 'tmp_fc.pth'))
                        z_updated = unrolled_optimizer(z, inputs, targets)
                        outputs = self.student_fc(z_updated)

                        loss = self.loss_fn(outputs, targets)

                        # grad = torch.autograd.grad(loss, student_parameters)

                        loss.backward()

                        student_optim.step()
                        # student_scheduler.step()

                        train_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += targets.size(0)
                        correct += predicted.eq(targets.data).cpu().sum()

                        progress_bar(batch_idx, len(self.loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

                        self.step = self.step + 1
                        self.adjust_learning_rate(student_optim, self.step)

                        batch_idx = batch_idx + 1

                    print('Epoch: %d | Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)' % (epoch, train_loss/(batch_idx+1), 100.*correct/total, correct, total))

                acc, test_loss = self.test(self.student, self.student_fc, test_loader=self.test_loader, epoch=epoch)
                res_student.append(acc)
                res_loss_student.append(test_loss)

                with open(logname, 'a') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    logwriter.writerow([epoch, acc])

            if self.visualize == False:
                fig = plt.figure()
                plt.plot(res_student, c="r", label="Student")
                plt.plot(res_example, c="g", label="SGD")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.legend()
                plt.show()

        if self.opt.train_baseline == False:
            # student
            self.opt.experiment = "Baseline"
            print("Start training {} ...".format(self.opt.experiment))
            logname = os.path.join(self.opt.log_path, 'results' + '_' + self.opt.experiment + '_' + self.opt.model + '_' + str(self.opt.seed) + '_' + self.opt.data_mode + '_' + str(self.opt.epsilon) + '.csv')
            if not os.path.exists(logname):
                with open(logname, 'w') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    logwriter.writerow(['epoch', 'test acc'])

            res_mixup = []
            res_loss_mixup = []

            tmp_student = networks.FullLayer(feature_dim=self.teacher.feature_num, n_classes=self.opt.n_classes).cuda()

            unrolled_optimizer = blackbox_implicit.UnrolledBlackBoxOptimizer(opt=self.opt, loader=self.loader, fc=tmp_student)

            res_student = []
            res_loss_student = []
            cls_loss = []

            w_init = self.student.state_dict()
            # new_weight = w_init
            train_loss = []

            self.step = 0
            self.best_acc = 0
            self.best_test_loss = 0
            self.init_train_loss = 0
            self.init_test_loss = 0
            self.init_feat_sim = 0

            # loader_eval = iter(self.val_loader)
            avg_train_loss = 0
            tmp_train_loss = 0
            feat_sim = 0

            self.baseline.load_state_dict(torch.load(os.path.join(self.opt.log_path, 'teacher_w0.pth')))
            self.baseline_fc.load_state_dict(torch.load(os.path.join(self.opt.log_path, 'teacher_fc_w0.pth')))
            # baseline_optim = torch.optim.SGD([{'params': self.baseline.parameters()}, {'params': self.baseline_fc.parameters()}],
            #                                 lr=self.opt.lr,
            #                                 momentum=self.opt.momentum, nesterov=self.opt.nesterov,
            #                                 weight_decay=self.opt.weight_decay)
            baseline_optim = torch.optim.SGD([{'params': self.baseline.parameters()}, {'params': self.baseline_fc.parameters()}], lr=0.001)
            # student_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(student_optim, len(self.loader)*self.opt.n_epochs)

            # student_parameters = list(self.student.parameters()) + list(self.student_fc.parameters())
            # train_loss = []
            # train_loss2 = []
            for epoch in range(self.opt.n_epochs):
                if epoch != 0:
                    self.baseline.train()
                    self.baseline_fc.train()
                    batch_idx = 0

                    train_loss = 0
                    correct = 0
                    total = 0

                    for (inputs, targets) in self.loader:

                        inputs, targets = inputs.cuda(), targets.long().cuda()
                        baseline_optim.zero_grad()

                        # model_mdl = copy.deepcopy(self.student)
                        z = self.baseline(inputs)

                        z_updated = unrolled_optimizer.forward_random(z)
                        outputs = self.baseline_fc(z_updated)

                        loss = self.loss_fn(outputs, targets)

                        # grad = torch.autograd.grad(loss, student_parameters)

                        loss.backward()

                        baseline_optim.step()
                        # student_scheduler.step()

                        train_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += targets.size(0)
                        correct += predicted.eq(targets.data).cpu().sum()

                        progress_bar(batch_idx, len(self.loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

                        self.step = self.step + 1
                        self.adjust_learning_rate(baseline_optim, self.step)

                        batch_idx = batch_idx + 1

                    print('Epoch: %d | Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)' % (epoch, train_loss/(batch_idx+1), 100.*correct/total, correct, total))

                acc, test_loss = self.test(self.baseline, self.baseline_fc, test_loader=self.test_loader, epoch=epoch)
                res_student.append(acc)
                res_loss_student.append(test_loss)

                with open(logname, 'a') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    logwriter.writerow([epoch, acc])

            if self.visualize == False:
                fig = plt.figure()
                plt.plot(res_student, c="r", label="Student")
                plt.plot(res_example, c="g", label="SGD")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.legend()
                plt.show()

        sys.exit()

        if self.visualize == False:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(14, 5.8)
            # a, b = plot_classifier(self.teacher, X.max(axis=0), X.min(axis=0))
            ax1.plot(res_student, 'r', label="Mixup Trained")
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

            ax2.plot(res_loss_student, c='r', label="Implicit Trained")
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

    def normalize_lp_norms(self, x, p):
        # norms = torch.flatten(x).norms.lp(p=p, axis=-1)
        norms = x.norm(p=p, dim=-1)
        eps = torch.tensor(1e-12)
        norms = torch.maximum(norms, eps)  # avoid divsion by zero
        factor = 1 / norms
        # factor = atleast_kd(factor, x.ndim)
        factor = factor.unsqueeze(1)
        return x * factor

    def clip_lp_norms(self, x, norm, p):
        norm = torch.tensor(norm).unsqueeze(0).cuda()

        # norms = torch.flatten(x).norms.lp(p=p, axis=-1)
        norms = x.norm(p=p, dim=-1)
        eps = torch.tensor(1e-12)
        norms = torch.maximum(norms, eps)  # avoid divsion by zero

        factor = torch.minimum(torch.ones(1).cuda(), norm / norms)  # clipping -> decreasing but not increasing
        # factor = atleast_kd(factor, x.ndim)
        factor = factor.unsqueeze(1)
        return x * factor

    # def project(self, x, x0, epsilon):
    #     return x0 + self.clip_lp_norms(x - x0, norm=epsilon, p=1)

    def project(self, x, x0, epsilon, p):
        return x0 + self.clip_lp_norms(x - x0, norm=epsilon, p=p)

    def projected_gradient_descent(self, model, fc, inputs, targets):
        # https://github.com/bethgelab/foolbox

        """Run the style transfer."""
        # print('Building the style transfer model..')

        optim = torch.optim.SGD([{'params': model.parameters()}, {'params': fc.parameters()}], lr=0.001, momentum=0.9, weight_decay=self.opt.decay)
        optim_loss = []

        eps = 1
        eps_batch = torch.ones(self.opt.batch_size) * eps
        eps_batch = eps_batch.cuda()

        model_orig = copy.deepcopy(model)

        pdist = torch.nn.PairwiseDistance(p=2)
        num_steps = 10
        step_size = 0.001
        epsilon = 0.1
        norm = 0.0
        p = 2

        z0 = model(inputs)
        z = z0

        optim_loss = []

        optim.zero_grad()
        for _ in range(num_steps):
            output = fc(z)
            loss = self.loss_fn(output, targets)
            gradients = torch.autograd.grad(outputs=loss,
                                            inputs=z,
                                            create_graph=True, retain_graph=True)

            gradients = self.normalize_lp_norms(gradients[0], p=p)
            z = z - step_size * gradients
            z = self.project(z, z0, epsilon, p)

            optim_loss.append(loss.item())

        # diff = pdist(z, z0)
        # print('diff', diff.max())

        # mask = (diff < eps).float().unsqueeze(1)

        # output = fc(z)
        # loss = self.loss_fn(output, targets)
        # loss.backward()
        # optim.step()

        # diff2 = torch.norm(z - z_new1, p='fro', dim=1) ** 2

        # x_proj = mask * x + (1 - mask) * proj * torch.sign(x)

        # offset = project_onto_l1_ball(diff, 0.1)
        # z_proj = z + offset

        # fig = plt.figure()
        # plt.plot(optim_loss, c="b", label="Mixup")
        # plt.xlabel("Epoch")
        # plt.ylabel("Accuracy")
        # plt.legend()
        # plt.show()

        # diff = z_org - z_tmp
        # z = z - diff

        return z, optim_loss

    def teach_linear_classifier(self, model, fc, inputs, targets):
        # https://github.com/bethgelab/foolbox

        """Run the style transfer."""
        # print('Building the style transfer model..')

        optim = torch.optim.SGD([{'params': model.parameters()}, {'params': fc.parameters()}], lr=0.001, momentum=0.9, weight_decay=self.opt.decay)
        optim_loss = []

        eps = 1
        eps_batch = torch.ones(self.opt.batch_size) * eps
        eps_batch = eps_batch.cuda()

        model_orig = copy.deepcopy(model)

        pdist = torch.nn.PairwiseDistance(p=2)
        num_steps = 10
        step_size = 0.001
        epsilon = 0.1
        norm = 0.0
        p = 2

        z0 = model(inputs)
        z = z0

        optim_loss = []

        optim.zero_grad()
        for _ in range(num_steps):
            output = fc(z)
            loss = self.loss_fn(output, targets)
            gradients = torch.autograd.grad(outputs=loss,
                                            inputs=z,
                                            create_graph=True, retain_graph=True)

            gradients = self.normalize_lp_norms(gradients[0], p=p)
            z = z - step_size * gradients
            z = self.project(z, z0, epsilon, p)

            optim_loss.append(loss.item())

        # diff = pdist(z, z0)
        # print('diff', diff.max())

        # mask = (diff < eps).float().unsqueeze(1)

        # output = fc(z)
        # loss = self.loss_fn(output, targets)
        # loss.backward()
        # optim.step()

        # diff2 = torch.norm(z - z_new1, p='fro', dim=1) ** 2

        # x_proj = mask * x + (1 - mask) * proj * torch.sign(x)

        # offset = project_onto_l1_ball(diff, 0.1)
        # z_proj = z + offset

        fig = plt.figure()
        plt.plot(optim_loss, c="b", label="Mixup")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

        # diff = z_org - z_tmp
        # z = z - diff

        return z

    def get_activation(self, name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    def data_sampler(self, X, y, i):
        i_min = i * self.opt.batch_size
        i_max = (i + 1) * self.opt.batch_size

        x = X[i_min:i_max].cuda()
        y = y[i_min:i_max].cuda()

        return x, y

    def query_model1(self):
        classes = torch.combinations(torch.arange(self.opt.n_query_classes))
        feat_sim = torch.empty(len(classes))

        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        '''
        for i, cls in enumerate(classes):
            a, b = cls[0], cls[1]

            _ = self.student(self.query_set[a, :])
            act1 = activation['latent'].squeeze()
            # act1 = activation['latent'].mean(0).squeeze()

            # act1_norm = (act1 - act1.mean(0)) / act1.std(0)

            _ = self.student(self.query_set[b, :])
            act2 = activation['latent'].squeeze()

            # act2 = activation['latent'].mean(0).squeeze()

            # act2_norm = (act2 - act2.mean(0)) / act2.std(0)

            # feat_sim[i] = HSIC(act1_norm, act2_norm)
            cos_sim = cos(act1, act2)
            feat_sim[i] = torch.mean(cos_sim)
        '''
        _ = self.student(self.query_set_1)
        act1 = activation['latent'].squeeze()
        _ = self.student(self.query_set_2)
        act2 = activation['latent'].squeeze()
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        feat_sim = cos(act1, act2)

        return feat_sim.cuda()


    def query_model(self):
        classes = torch.combinations(torch.arange(self.opt.n_query_classes))
        feat_sim = torch.empty(len(classes))

        cos = nn.CosineSimilarity(dim=0, eps=1e-6)

        m = nn.BatchNorm1d(512, affine=False).cuda()
        '''
        for i, cls in enumerate(classes):
            a, b = cls[0], cls[1]

            _ = self.student(self.query_set[a, :])
            act1 = activation['latent'].squeeze()
            # act1 = activation['latent'].mean(0).squeeze()

            # act1_norm = (act1 - act1.mean(0)) / act1.std(0)

            _ = self.student(self.query_set[b, :])
            act2 = activation['latent'].squeeze()

            # act2 = activation['latent'].mean(0).squeeze()

            # act2_norm = (act2 - act2.mean(0)) / act2.std(0)

            # feat_sim[i] = HSIC(act1_norm, act2_norm)
            cos_sim = cos(act1, act2)
            feat_sim[i] = torch.mean(cos_sim)
        '''
        _ = self.student(self.query_set_1)
        act1 = activation['latent'].squeeze()
        # act1_norm = m(act1)
        _ = self.student(self.query_set_2)
        act2 = activation['latent'].squeeze()
        # act2_norm = m(act2)

        # c = act1_norm.T @ act2_norm
        # c.div_(self.opt.n_query_classes)

        # on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        # off_diag = off_diagonal(c).pow_(2).sum()
        # feat_sim_loss = on_diag + 0.0051 * off_diag

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        feat_sim = cos(act1, act2)

        return feat_sim.cuda()


    def query_model3(self):
        classes = torch.combinations(torch.arange(self.opt.n_query_classes))
        feat_sim = torch.empty(len(classes))

        # m = nn.BatchNorm1d(512, affine=False).cuda()

        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        for i, cls in enumerate(classes):
            a, b = cls[0], cls[1]

            _ = self.student(self.query_set[a, :])
            act1 = activation['latent'].squeeze()
            # act1_norm = m(act1)
            # act1 = activation['latent'].mean(0).squeeze()

            _ = self.student(self.query_set[b, :])
            act2 = activation['latent'].squeeze()
            # act2_norm = m(act2)
            # act2 = activation['latent'].mean(0).squeeze()

            # feat_sim[i] = HSIC(act1_norm, act2_norm)
            cos_sim = cos(act1, act2)
            feat_sim[i] = torch.mean(cos_sim)


        # _ = self.student(self.query_set_1)
        # act1 = activation['latent'].squeeze()
        # _ = self.student(self.query_set_2)
        # act2 = activation['latent'].squeeze()
        # cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        # feat_sim = cos(act1, act2)
        print(feat_sim)
        return feat_sim.cuda()

    def train(self, model, fc, train_loader, loss_fn, optimizer, epoch):
        model.train()
        fc.train()
        for batch_idx, (data, target) in enumerate(train_loader):

            # first_image = np.array(data.cpu(), dtype='float')
            # pixels = first_image.reshape((28, 28))
            # plt.imshow(pixels, cmap='gray')
            # plt.title("Label {}".format(target.item()))
            # plt.show()

            data, target = data.cuda(), target.long().cuda()
            optimizer.zero_grad()
            z = model(data)
            output = fc(z)
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

    def test(self, model, fc, test_loader, epoch, netG=None):
        model.eval()
        fc.eval()
        test_loss = 0
        correct = 0
        loss_fn = nn.CrossEntropyLoss(reduction='sum')
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.cuda(), target.cuda()
                z = model(data)
                output = fc(z)

                test_loss += loss_fn(output, target.long()).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        acc = correct / len(test_loader.dataset)
        self.log(mode="test", name="acc", value=acc, step=epoch)

        print('\nEpoch: {}, Test Average loss: {:.4f}, Test Accuracy: {}/{} ({:.0f}%)\n'.format(
            epoch, test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        self.log(mode="test", name="loss", value=test_loss, step=epoch)

        # if epoch == 0 or acc > self.best_acc:
        self.save_model(model=model, name=self.opt.experiment)
        if netG is not None:
            self.save_model(model=netG, name='netG_{}'.format(epoch))
        if acc > self.best_acc:
            best_acc = acc
        if self.best_test_loss > test_loss:
            self.best_test_loss = test_loss

        return acc, test_loss

    def avg_loss(self, model, data_loader):
        model.eval()
        train_loss = 0
        correct = 0
        loss_fn = nn.CrossEntropyLoss(reduction='sum')
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.cuda(), target.cuda()
                output = model(data)

                train_loss += loss_fn(output, target.long()).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        train_loss /= len(data_loader.dataset)

        return train_loss

    def model_features(self, train_loss, epoch):
        current_iter = self.step / (self.opt.n_epochs * len(self.train_loader))

        if epoch == 1:
            avg_training_loss = 1.0
        else:
            avg_training_loss = train_loss / self.init_train_loss

        if epoch == 1:
            best_val_loss = 1.0
        else:
            best_val_loss = self.best_test_loss / self.init_test_loss
        model_features = [current_iter, avg_training_loss, best_val_loss]
        return torch.FloatTensor(model_features).cuda()

    def make_results_img(self, X, Y, generated_samples, generated_labels, w_diff_example, w_diff_baseline, w_diff_student, loss_student, loss_g, loss_d, epoch, proj_matrix):
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
