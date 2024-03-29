from __future__ import absolute_import, division, print_function

import numpy as np
import time
import json

import sys

import csv

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
import teachers.omniscient_teacher as omniscient
import teachers.utils as utils
import matplotlib.pyplot as plt

import networks.blackbox_mixup_cnn as blackbox_mixup

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


class Trainer:
    def __init__(self, options):
        self.opt = options

        self.opt.model_name = "blackbox_mixup_cnn_" + self.opt.data_mode

        self.opt.log_path = os.path.join(CONF.PATH.LOG, self.opt.model_name, str(self.opt.seed), str(self.opt.model), str(self.opt.experiment))
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

            self.train_loader = DataLoader(self.train_dataset, batch_size=self.opt.batch_size, num_workers=self.opt.num_workers, pin_memory=True, shuffle=True)
            self.val_loader = DataLoader(self.val_dataset, batch_size=self.opt.batch_size, num_workers=self.opt.num_workers, pin_memory=True, shuffle=True)
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.opt.batch_size, num_workers=self.opt.num_workers, pin_memory=True, shuffle=False)

            self.loader = DataLoader(dataset, batch_size=self.opt.batch_size, num_workers=self.opt.num_workers, pin_memory=True, shuffle=True)

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
            print(CONF.PATH.DATA)
            dataset = torchvision.datasets.CIFAR100(root=CONF.PATH.DATA, train=True, download=True, transform=transform_train)
            self.test_dataset = torchvision.datasets.CIFAR100(root=CONF.PATH.DATA, train=False, download=True, transform=transform_test)

            self.train_dataset, self.val_dataset = random_split(dataset, [40000, 10000])

            self.train_loader = DataLoader(self.train_dataset, batch_size=self.opt.batch_size, num_workers=self.opt.num_workers, pin_memory=True, shuffle=True, drop_last=True)
            self.val_loader = DataLoader(self.val_dataset, batch_size=self.opt.batch_size, num_workers=self.opt.num_workers, pin_memory=True, shuffle=True, drop_last=True)
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.opt.batch_size, num_workers=self.opt.num_workers, pin_memory=True, shuffle=False, drop_last=True)

            self.loader = DataLoader(dataset, batch_size=self.opt.batch_size, num_workers=self.opt.num_workers, pin_memory=True, shuffle=True, drop_last=True)

        elif self.opt.data_mode == "mnist":
            # MNIST normalizing
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
            self.train_dataset = torchvision.datasets.MNIST(root=CONF.PATH.DATA, train=True, download=True, transform=transform)
            # train, valid = random_split(train_dataset, [50000, 10000])
            self.test_dataset = torchvision.datasets.MNIST(root=CONF.PATH.DATA, train=False, download=True, transform=transform)
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.opt.batch_size, shuffle=True, num_workers=self.opt.num_workers, pin_memory=True)
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.opt.batch_size, shuffle=True, num_workers=self.opt.num_workers, pin_memory=True)

        elif self.opt.data_mode == "fashion_mnist":
            # MNIST normalizing
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
            dataset = torchvision.datasets.FashionMNIST(root=CONF.PATH.DATA, train=True, download=True, transform=transform)
            # train, valid = random_split(train_dataset, [50000, 10000])
            self.test_dataset = torchvision.datasets.FashionMNIST(root=CONF.PATH.DATA, train=False, download=True, transform=transform)

            self.train_dataset, self.val_dataset = random_split(dataset, [50000, 10000])

            self.train_loader = DataLoader(self.train_dataset, batch_size=self.opt.batch_size, shuffle=True, num_workers=self.opt.num_workers, pin_memory=True)
            self.val_loader = DataLoader(self.val_dataset, batch_size=self.opt.batch_size, shuffle=True, num_workers=self.opt.num_workers, pin_memory=True)
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.opt.batch_size, shuffle=True, num_workers=self.opt.num_workers, pin_memory=True)

            self.loader = DataLoader(dataset, batch_size=self.opt.batch_size, num_workers=self.opt.num_workers, pin_memory=True, shuffle=True, drop_last=False)

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

        self.query_set_1, self.query_set_2 = self.get_query_set()
        # self.query_set = self.get_query_set()
        # features, labels = self.get_query_set()

    '''
    def get_teacher_student(self):
        self.teacher = networks.ResNet18(in_channels=self.opt.channels, num_classes=self.opt.n_classes).cuda()
        self.teacher.apply(initialize_weights)
        self.teacher_fc = networks.FullLayer(feature_dim=512, n_classes=self.opt.n_classes).cuda()
        # torch.save(self.teacher.state_dict(), 'teacher_w0.pth')
        self.teacher_fc.apply(initialize_weights)
        # torch.save(self.teacher_fc.state_dict(), 'teacher_fc_w0.pth')

        # path = os.path.join(self.opt.log_path, 'weights/best_model_SGD.pth')

        self.baseline = networks.ResNet18(in_channels=self.opt.channels, num_classes=self.opt.n_classes).cuda()
        self.baseline_fc = networks.FullLayer(feature_dim=512, n_classes=self.opt.n_classes).cuda()
        # self.baseline.load_state_dict(torch.load(path))
        # self.baseline.model.avgpool.register_forward_hook(self.get_activation('latent'))
        self.baseline = networks.ResNet18(in_channels=self.opt.channels, num_classes=self.opt.n_classes).cuda()
        self.baseline_fc = networks.FullLayer(feature_dim=512, n_classes=self.opt.n_classes).cuda()

        self.baseline.load_state_dict(self.teacher.state_dict())
        self.baseline.load_state_dict(self.teacher.state_dict())
    '''

    def get_teacher_student(self):
        if self.opt.model == "NET":
            self.teacher = networks.NET(n_in=self.opt.n_in, in_channels=self.opt.channels, num_classes=self.opt.n_classes).cuda()
        else:
            self.teacher = networks.CNN(self.opt.model, in_channels=self.opt.channels, num_classes=self.opt.n_classes).cuda()
        self.teacher.apply(initialize_weights)
        torch.save(self.teacher.state_dict(), os.path.join(self.opt.log_path, 'teacher_w0.pth'))

        # path = os.path.join(self.opt.log_path, 'weights/best_model_SGD.pth')

        if self.opt.model == "NET":
            self.student = networks.NET(n_in=self.opt.n_in, in_channels=self.opt.channels, num_classes=self.opt.n_classes).cuda()
        else:
            self.student = networks.CNN(self.opt.model, in_channels=self.opt.channels, num_classes=self.opt.n_classes).cuda()
        # self.baseline.load_state_dict(torch.load(path))
        # self.baseline.model.avgpool.register_forward_hook(self.get_activation('latent'))
        if self.opt.model == "NET":
            self.baseline = networks.NET(n_in=self.opt.n_in, in_channels=self.opt.channels, num_classes=self.opt.n_classes).cuda()
        else:
            self.baseline = networks.CNN(self.opt.model, in_channels=self.opt.channels, num_classes=self.opt.n_classes).cuda()

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
        if epoch >= 50: # 100
            # lr /= 10
            lr = 0.01
        if epoch >= 75: # 150
            # lr /= 100
            lr = 0.001
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
        query_set = torch.empty(self.opt.n_query_classes, self.opt.channels, self.opt.img_size, self.opt.img_size)

        val_iter = iter(self.val_loader)

        try:
            (inputs, targets) = val_iter.next()
        except:
            val_iter = iter(self.val_loader)
            (inputs, targets) = val_iter.next()

        features = self.baseline(inputs.cuda())
        outputs = self.baseline_fc(features.cuda())

        # features = activation['latent']

        return features.cuda(), targets.cuda()

    def get_query_set2(self):
        """decrease the learning rate at 100 and 150 epoch"""
        n_samples = 10
        query_set = torch.zeros(self.opt.n_query_classes, n_samples, self.opt.channels, self.opt.img_size, self.opt.img_size)

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

        print(self.opt)

        print("Training")

        if self.opt.model == "NET":
            tmp_student = networks.NET(n_in=self.opt.n_in, in_channels=self.opt.channels, num_classes=self.opt.n_classes).cuda()
        else:
            tmp_student = networks.CNN(self.opt.model, in_channels=self.opt.channels, num_classes=self.opt.n_classes).cuda()

        if self.opt.experiment == "First_Order_Optimization":
            # mixup student
            self.opt.experiment = "First_Order_Optimization"
            print("Start training {} ...".format(self.opt.experiment))
            logname = os.path.join(self.opt.log_path, 'results' + '_' + self.opt.experiment + '_' + str(self.opt.seed) + '.csv')
            if not os.path.exists(logname):
                with open(logname, 'w') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    logwriter.writerow(['epoch', 'test acc'])

            res_mixup = []
            res_loss_mixup = []

            netG = blackbox_mixup.Generator(self.opt).cuda()
            netG.apply(weights_init)

            optimizer_G = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999), eps=1e-08, amsgrad=False)
            # optimizer_G = torch.optim.SGD(netG.parameters(), lr=0.0002)
            unrolled_optimizer = blackbox_mixup.UnrolledBlackBoxOptimizer(opt=self.opt, teacher=self.teacher, student=tmp_student, generator=netG, train_dataset=self.train_dataset, val_loader=self.val_loader, proj_matrix=None)
            res_student = []
            res_loss_student = []
            cls_loss = []

            w_init = self.baseline.state_dict()
            # new_weight = w_init
            train_loss = 0.0

            self.baseline.load_state_dict(torch.load(os.path.join(self.opt.log_path, 'teacher_w0.pth')))
            baseline_optim = torch.optim.Adam(self.baseline.parameters(), lr=self.opt.lr)

            self.step = 0
            self.best_acc = 0
            self.best_test_loss = 1000
            self.init_train_loss = 0
            self.init_test_loss = 0
            self.init_feat_sim = 0

            loader_eval = iter(self.val_loader)
            avg_train_loss = 0
            tmp_train_loss = 0
            feat_sim = 0

            model_features = torch.FloatTensor([0, 1.0, 1.0]).cuda()

            # optimizer = torch.optim.SGD(netG.model.parameters(), lr=self.opt.lr, momentum=0.9, weight_decay=self.opt.decay)
            for epoch in tqdm(range(self.opt.n_epochs)):
                if epoch != 0:
                    self.baseline.train()
                    for batch_idx, (inputs, targets) in enumerate(self.train_loader):

                        # ---------------------
                        #  Train Generator
                        # ---------------------

                        netG.zero_grad()
                        w_t = netG.state_dict()
                        gradients, generator_loss, G_loss = unrolled_optimizer(w_t, model_features)

                        with torch.no_grad():
                            for p, g in zip(netG.model.parameters(), gradients):
                                p.grad = g

                        # outputs_normal = self.baseline(val_inputs)
                        # loss_normal = self.loss_fn(outputs_normal, val_targets)

                        # tmp_train_loss = tmp_train_loss + loss.item()

                        # optimizer_G.zero_grad()
                        # loss.backward()
                        optimizer_G.step()
                        netG.eval()

                        # ---------------------
                        #  Train Student
                        # ---------------------

                        inputs, targets = inputs.cuda(), targets.long().cuda()

                        targets_onehot = torch.FloatTensor(inputs.shape[0], self.opt.n_classes).cuda()
                        targets_onehot.zero_()
                        targets_onehot.scatter_(1, targets.unsqueeze(1), 1)

                        #  model_features = self.model_features(avg_train_loss, epoch)

                        # lam = np.random.beta(1.0, 1.0)
                        # lam = torch.tensor(lam, dtype=torch.float).cuda()
                        # norm_offset = netG(inputs, targets.long(), model_features, lam)
                        # norm_offset = torch.max(norm_offset, dim=1).indices
                        # offset_count = torch.bincount(norm_offset)

                        # if torch.argmax(offset_count) == 0:
                        #     offset = 0.1
                        # elif torch.argmax(offset_count) == 2:
                        #     offset = - 0.1
                        # else:
                        #     offset = 0

                        # lam = lam + offset
                        # lam = torch.clamp(lam, min=0, max=1)

                        index = torch.randperm(inputs.shape[0]).cuda()

                        inputs_logits_1 = self.baseline(inputs)
                        inputs_logits_2 = self.baseline(inputs[index, :])
                        lam = netG(inputs_logits_1, inputs_logits_2, targets, targets[index], model_features)
                        # lam = netG(inputs, inputs[index, :], targets, targets[index], model_features)

                        self.baseline.train()

                        x_lam = torch.reshape(lam, (inputs.shape[0], 1, 1, 1)).cuda()
                        y_lam = torch.reshape(lam, (inputs.shape[0], 1)).cuda()

                        mixed_x = x_lam * inputs + (1 - x_lam) * inputs[index, :]
                        mixed_y = y_lam * targets_onehot + (1 - y_lam) * targets_onehot[index]

                        outputs = self.baseline(mixed_x)

                        loss_stu = self.loss_fn(outputs, mixed_y)
                        # loss_stu = lam * self.loss_fn(outputs, targets_a) + (1 - lam) * self.loss_fn(outputs, targets_b)

                        train_loss = train_loss + loss_stu.item()

                        baseline_optim.zero_grad()
                        loss_stu.backward()
                        baseline_optim.step()
                        self.baseline.eval()

                        self.step += 1
                        if batch_idx % self.opt.log_interval == 0:
                            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                epoch, batch_idx * len(inputs), len(self.train_loader.dataset),
                                100. * batch_idx / len(self.train_loader), loss_stu.item()), '\t')
                            self.log(mode="train", name="loss_student", value=loss_stu.item(), step=self.step)
                            # self.log(mode="val", name="loss_teacher", value=loss.item(), step=self.step)

                        if self.step == 1:
                            _, _ = self.val(self.baseline, val_loader=self.val_loader, epoch=epoch)

                            self.init_train_loss = train_loss / self.step
                            avg_train_loss = self.init_train_loss
                            self.init_test_loss = self.best_test_loss
                            model_features = self.model_features(avg_train_loss)

                        else:
                            avg_train_loss = train_loss / self.step
                            model_features = self.model_features(avg_train_loss)

                        if self.step % 100 == 0:
                            # _, _ = self.test(self.baseline, test_loader=self.test_loader, epoch=epoch)
                            _, _ = self.val(self.baseline, val_loader=self.val_loader, epoch=epoch)

                # _, _ = self.val(self.baseline, val_loader=self.val_loader, epoch=epoch, netG=netG, save=True)
                acc, test_loss = self.test(self.baseline, test_loader=self.test_loader, epoch=epoch, netG=netG)
                res_student.append(acc)
                res_loss_student.append(test_loss)

                with open(logname, 'a') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    logwriter.writerow([epoch, acc])

            torch.save(netG.state_dict(), os.path.join(self.opt.log_path, 'weights/best_model_netG.pth'))

        if self.opt.experiment == "Second_Order_Optimization":
            # mixup student
            self.opt.experiment = "Second_Order_Optimization"
            print("Start training {} ...".format(self.opt.experiment))
            logname = os.path.join(self.opt.log_path, 'results' + '_' + self.opt.experiment + '_' + str(self.opt.seed) + '.csv')
            if not os.path.exists(logname):
                with open(logname, 'w') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    logwriter.writerow(['epoch', 'test acc'])

            res_mixup = []
            res_loss_mixup = []

            netG = blackbox_mixup.Generator(self.opt).cuda()
            netG.apply(weights_init)

            # optimizer_G = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999), eps=1e-08, amsgrad=False)
            # optimizer_G = torch.optim.SGD(netG.parameters(), lr=0.0002)
            unrolled_optimizer = blackbox_mixup.UnrolledBlackBoxOptimizer(opt=self.opt, teacher=self.teacher, student=self.baseline, generator=netG, train_dataset=self.train_dataset, val_loader=self.val_loader, proj_matrix=None)
            res_student = []
            res_loss_student = []
            cls_loss = []

            w_init = self.baseline.state_dict()
            # new_weight = w_init
            train_loss = 0.0

            self.baseline.load_state_dict(torch.load(os.path.join(self.opt.log_path, 'teacher_w0.pth')))
            baseline_optim = torch.optim.Adam(self.baseline.parameters(), lr=self.opt.lr)

            self.step = 0
            self.best_acc = 0
            self.best_test_loss = 1000
            self.init_train_loss = 0
            self.init_test_loss = 0
            self.init_feat_sim = 0

            loader_eval = iter(self.val_loader)
            avg_train_loss = 0
            tmp_train_loss = 0
            feat_sim = 0

            model_features = torch.FloatTensor([0, 1.0, 1.0]).cuda()

            for epoch in tqdm(range(self.opt.n_epochs)):
                if epoch != 0:
                    self.baseline.train()
                    for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                        self.step = self.step + 1

                        try:
                            (val_inputs, val_targets) = loader_eval.next()
                        except:
                            loader_eval = iter(self.val_loader)
                            (val_inputs, val_targets) = loader_eval.next()

                        # ---------------------
                        #  Generator Input
                        # ---------------------

                        # self.baseline.eval()

                        val_inputs, val_targets = val_inputs.cuda(), val_targets.long().cuda()

                        # ---------------------
                        #  Student Input
                        # ---------------------

                        inputs, targets = inputs.cuda(), targets.long().cuda()

                        loss = unrolled_optimizer.step(inputs, targets, val_inputs, val_targets, self.opt.lr, baseline_optim, model_features)

                        train_loss = train_loss + loss

                        if batch_idx % self.opt.log_interval == 0:
                            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                epoch, batch_idx * len(inputs), len(self.train_loader.dataset),
                                100. * batch_idx / len(self.train_loader), loss), '\t',
                                'Val Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                epoch, batch_idx * len(inputs), len(self.val_loader.dataset),
                                100. * batch_idx / len(self.val_loader), loss))
                            self.log(mode="train", name="loss_student", value=loss, step=self.step)
                            # self.log(mode="val", name="loss_teacher", value=loss.item(), step=self.step)

                # _, _ = self.val(self.baseline, val_loader=self.val_loader, epoch=epoch, netG=netG, save=True)
                acc, test_loss = self.test(self.baseline, test_loader=self.test_loader, epoch=epoch, netG=netG)
                res_student.append(acc)
                res_loss_student.append(test_loss)

                with open(logname, 'a') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    logwriter.writerow([epoch, acc])

            torch.save(netG.state_dict(), os.path.join(self.opt.log_path, 'weights/best_model_netG.pth'))

            if self.visualize == False:
                fig = plt.figure()
                # plt.plot(w_diff_mixup, c="c", label="Mixup")
                # plt.plot(res_example, c="g", label="SGD")
                plt.plot(res_mixup, c="b", label="Mixup")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.legend()
                plt.show()

        netG_path = os.path.join(self.opt.log_path, 'weights/best_model_netG.pth')
        netG = blackbox_mixup.Generator(self.opt).cuda()
        netG.load_state_dict(torch.load(netG_path))

        self.opt.experiment = "Student"
        print("Start training {} ...".format(self.opt.experiment))
        logname = os.path.join(self.opt.log_path, 'results' + '_' + self.opt.experiment + '_' + str(self.opt.seed) + '.csv')
        if not os.path.exists(logname):
            with open(logname, 'w') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow(['epoch', 'test acc'])

        res_student = []
        res_loss_student = []

        self.student.load_state_dict(torch.load(os.path.join(self.opt.log_path, 'teacher_w0.pth')))
        student_optim = torch.optim.Adam(self.student.parameters(), lr=self.opt.lr)

        self.step = 0
        self.best_acc = 0
        self.best_test_loss = 1000
        self.init_train_loss = 0
        self.init_test_loss = 0
        self.init_feat_sim = 0
        avg_train_loss = 0
        tmp_train_loss = 0

        train_loss = 0.0

        model_features = torch.FloatTensor([0, 1.0, 1.0]).cuda()

        for epoch in tqdm(range(self.opt.n_epochs)):
            if epoch != 0:
                for batch_idx, (inputs, targets) in enumerate(self.loader):
                    n_samples = inputs.shape[0]
                    inputs, targets = inputs.cuda(), targets.long().cuda()
                    # mixed_x, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=1.0)
                    targets_onehot = torch.FloatTensor(n_samples, self.opt.n_classes).cuda()
                    targets_onehot.zero_()
                    targets_onehot.scatter_(1, targets.unsqueeze(1), 1)

                    # alpha = np.random.beta(1.0, 1.0)
                    # alpha = torch.tensor(alpha, dtype=torch.float).cuda()

                    # lam = torch.rand(n_samples, 1).cuda()

                    index = torch.randperm(inputs.shape[0]).cuda()

                    inputs_logits_1 = self.student(inputs)
                    inputs_logits_2 = self.student(inputs[index, :])
                    lam = netG(inputs_logits_1, inputs_logits_2, targets, targets[index], model_features)

                    # lam = netG(inputs, inputs[index, :], targets, targets[index], model_features)

                    self.student.train()
                    x_lam = torch.reshape(lam, (n_samples, 1, 1, 1))
                    y_lam = torch.reshape(lam, (n_samples, 1))

                    mixed_x = x_lam * inputs + (1 - x_lam) * inputs[index, :]
                    mixed_y = y_lam * targets_onehot + (1 - y_lam) * targets_onehot[index]

                    outputs = self.student(mixed_x)
                    loss = self.loss_fn(outputs, mixed_y)

                    # loss = lam * self.loss_fn(outputs, targets_a) + (1 - lam) * self.loss_fn(outputs, targets_b)
                    # print("loss2", loss)

                    # targets_a, targets_b = targets, targets[index]
                    # loss1 = mixup_criterion_batch(self.loss_fn, outputs, targets_a, targets_b, y_lam)
                    # print("loss1", loss1)

                    # loss = self.loss_fn(outputs, mixed_y.long())
                    # print("loss2", loss)
                    train_loss = train_loss + loss.item()

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

            # if epoch % 2 == 0:
            #    self.query_set_1, self.query_set_2 = self.get_query_set()

            with open(logname, 'a') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow([epoch, acc])

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

    def query_model(self):

        feat1 = self.baseline(self.query_set_1)
        act1 = feat1.detach().squeeze()
        # act1_norm = m(act1)
        feat2 = self.baseline(self.query_set_2)
        act2 = feat2.detach().squeeze()

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        feat_sim = cos(act1, act2)

        return feat_sim.cuda()

    def train(self, model, train_loader, loss_fn, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):

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

    def val(self, model, val_loader, epoch):
        model.eval()
        test_loss = 0
        correct = 0
        loss_fn = nn.CrossEntropyLoss(reduction='sum')

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.cuda(), target.cuda()
                output = model(data)

                test_loss += loss_fn(output, target.long()).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(val_loader.dataset)
        acc = correct / len(val_loader.dataset)

        # if epoch == 0 or acc > self.best_acc:
        #         self.save_model(model=model, name=self.opt.experiment)
        if acc > self.best_acc:
            self.best_acc = acc
        if self.best_test_loss > test_loss:
            self.best_test_loss = test_loss

        model.train()
        return acc, test_loss

    def test(self, model, test_loader, epoch, netG=None):
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

        print('\nEpoch: {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            epoch, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
        self.log(mode="test", name="loss", value=test_loss, step=epoch)

        if epoch == 0 or acc > self.best_acc:
            self.save_model(model=model, name=self.opt.experiment)
        # if acc > self.best_acc:
        #     self.best_acc = acc
        # if self.best_test_loss > test_loss:
        #     self.best_test_loss = test_loss

        model.train()
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

    def model_features(self, avg_train_loss):
        current_iter = self.step / (self.opt.n_epochs * len(self.train_loader))
        avg_training_loss = avg_train_loss / self.init_train_loss
        best_val_loss = self.best_test_loss / self.init_test_loss

        model_features = [current_iter, avg_training_loss, best_val_loss]

        return torch.FloatTensor(model_features).cuda()

    def save_model(self, model, name):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.opt.log_path, "weights")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        save_path = os.path.join(save_folder, "best_model_{}.pth".format(name))
        to_save = model.state_dict()
        torch.save(to_save, save_path)
