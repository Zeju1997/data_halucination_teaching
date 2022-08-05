# https://github.com/jakeoung/BayesianUnrolling/blob/master/unroll/model/vanilla.py

# from . import attention
# from . import softargmax

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

import numpy as np


class Generator(nn.Module):
    def __init__(self, opt, teacher, student):
        super(Generator, self).__init__()

        self.opt = opt
        self.label_emb = nn.Embedding(self.opt.n_classes, self.opt.label_dim)
        self.img_shape = (self.opt.channels, self.opt.img_size, self.opt.img_size)

        in_channels = teacher.lin.weight.size(1) + student.lin.weight.size(1) + self.opt.latent_dim + self.opt.label_dim

        def block(in_feat, out_feat, normalize=False):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # *block(self.opt.dim + self.opt.label_dim, 128, normalize=False),
            *block(in_channels, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels.to(torch.int64)), noise), -1)
        img = self.model(gen_input)
        # img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        self.opt = opt
        self.label_embedding = nn.Embedding(self.opt.n_classes, self.opt.label_dim)
        self.img_shape = (self.opt.channels, self.opt.img_size, self.opt.img_size)

        self.model = nn.Sequential(
            nn.Linear(opt.label_dim + int(np.prod(self.img_shape)), 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            # nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        # d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        d_in = torch.cat((img, self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity


class Generator_moon(nn.Module):
    def __init__(self, opt, teacher, student):
        super(Generator, self).__init__()

        self.opt = opt
        self.label_embedding = nn.Embedding(self.opt.n_classes, self.opt.label_dim)

        in_channels = teacher.lin.weight.size(1) + student.lin.weight.size(1) + self.opt.dim + self.opt.label_dim

        self.input_fc = nn.Linear(in_channels, self.opt.hidden_dim*4, bias=False)
        self.hidden_fc = nn.Linear(self.opt.hidden_dim*4, self.opt.hidden_dim*2, bias=False)
        self.output_fc = nn.Linear(self.opt.hidden_dim*2, self.opt.dim, bias=False)
        # self.activation = nn.LeakyReLU(0.1)
        self.activation = nn.ReLU()
        # self.activation = nn.Tanh()

    def forward(self, z, label):
        x = torch.cat((z, self.label_embedding(label.to(torch.int64))), dim=1)
        x = self.activation(self.input_fc(x))
        x = self.activation(self.hidden_fc(x))
        return self.output_fc(x)


class Discriminator_moon(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        self.opt = opt
        self.label_embedding = nn.Embedding(self.opt.n_classes, self.opt.label_dim)

        self.model = nn.Sequential(
            nn.Linear(opt.label_dim + self.opt.dim, self.opt.hidden_dim, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(self.opt.hidden_dim, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity


class UnrolledOptimizer(nn.Module):
    """
    Args:
        - nscale : number of scales
        - alpha : scale factor in the softmax in the expansion (rho in the paper)
        - nblock : number of stages (K in the paper)
        - K : kernel size
    """
    def __init__(self, opt, teacher, student, generator, X, y, proj_matrix):
        super(UnrolledOptimizer, self).__init__()

        self.opt = opt

        self.optim_blocks = nn.ModuleList()

        self.loss_fn = nn.MSELoss()
        # self.adversarial_loss = nn.BCELoss()
        self.adversarial_loss = nn.MSELoss()

        self.teacher = teacher
        self.student = student
        self.generator = generator

        self.X = X
        self.Y = y

        self.nb_batch = int(self.X.shape[0] / self.opt.batch_size)

        self.proj_matrix = proj_matrix

    def forward(self, weight, w_star, w_init, netD, valid):
        # self.generator.linear.weight = weight
        # self.student.lin.weight = w_init

        with torch.no_grad():
            # for param1 in self.generator.parameters():
            #    param1 = weight
            self.generator.load_state_dict(weight)
            for param1 in self.student.parameters():
                param1 = w_init
            for param2 in self.teacher.parameters():
                param2 = w_star

        loss_stu = 0
        w_loss = 0
        tau = 1

        new_weight = w_init

        model_paramters = list(self.generator.parameters())

        for i in range(self.opt.n_unroll_blocks):
            w_t = self.student.lin.weight

            i = torch.randint(0, self.nb_batch, size=(1,)).item()
            i_min = i * self.opt.batch_size
            i_max = (i + 1) * self.opt.batch_size

            # gt_x = self.X[i_min:i_max].cuda()
            y = self.Y[i_min:i_max].cuda()

            # Sample noise and labels as generator input
            # z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, gt_x.shape)))
            z = Variable(torch.randn((self.opt.batch_size, self.opt.latent_dim))).cuda()

            # x = torch.cat((w_t, w_t-w_star, gt_x, y.unsqueeze(0)), dim=1)
            x = torch.cat((w_t, w_t-w_star, z), dim=1)
            generated_x = self.generator(x, y)

            generated_x_proj = generated_x @ self.proj_matrix.cuda()

            # self.student.train()
            out = self.student(generated_x_proj)

            loss = self.loss_fn(out, y.float())
            grad = torch.autograd.grad(loss, self.student.lin.weight, create_graph=True)
            # new_weight = self.student.lin.weight - 0.001 * grad[0]
            new_weight = new_weight - 0.001 * grad[0]
            self.student.lin.weight = torch.nn.Parameter(new_weight.cuda())
            # self.student.lin.weight = self.student.lin.weight - 0.001 * grad[0].cuda()

            # tau = np.exp(-i / 0.95)
            if i != -1:
                tau = 1
            else:
                tau = 0.95 * tau

            # self.student.eval()
            out_stu = self.teacher(generated_x_proj)
            # out_stu = self.student(generated_x)
            loss_stu = loss_stu + tau * self.loss_fn(out_stu, y)

        w_loss = torch.linalg.norm(w_star - new_weight, ord=2) ** 2

        '''
        grad_stu = torch.autograd.grad(outputs=loss_stu,
                                       inputs=model_paramters,
                                       create_graph=True, retain_graph=True)



        grad_stu_w = torch.autograd.grad(outputs=w_loss,
                                       inputs=model_paramters,
                                       create_graph=True, retain_graph=True)
        '''

        # Loss measures generator's ability to fool the discriminator
        # valid = Variable(torch.cuda.FloatTensor(self.batch_size, 1).fill_(1.0), requires_grad=False)

        w_t = self.student.lin.weight

        i = torch.randint(0, self.nb_batch, size=(1,)).item()
        i_min = i * self.opt.batch_size
        i_max = (i + 1) * self.opt.batch_size

        # gt_x = self.X[i_min:i_max].cuda()
        generated_labels = self.Y[i_min:i_max].cuda()

        z = Variable(torch.randn((self.opt.batch_size, self.opt.latent_dim))).cuda()
        # z = Variable(torch.randn(gt_x.shape)).cuda()

        # x = torch.cat((w_t, w_t-w_star, gt_x, generated_labels.unsqueeze(0)), dim=1)
        x = torch.cat((w_t, w_t-w_star, z), dim=1)
        generated_samples = self.generator(x, generated_labels)

        # generated_labels = generated_labels.float()

        validity = netD(generated_samples, Variable(generated_labels.type(torch.cuda.LongTensor)))
        g_loss = self.adversarial_loss(validity, valid)

        loss_stu = loss_stu + w_loss + g_loss

        grad_stu = torch.autograd.grad(outputs=loss_stu,
                                       inputs=model_paramters,
                                       create_graph=True, retain_graph=True)

        return grad_stu, loss_stu, generated_samples, generated_labels, g_loss


class UnrolledOptimizer_moon(nn.Module):
    """
    Args:
        - nscale : number of scales
        - alpha : scale factor in the softmax in the expansion (rho in the paper)
        - nblock : number of stages (K in the paper)
        - K : kernel size
    """
    def __init__(self, nblocks, teacher, student, generator, X, y, batch_size):
        super(UnrolledOptimizer, self).__init__()

        self.nblock = nblocks
        self.optim_blocks = nn.ModuleList()

        self.loss_fn = nn.MSELoss()
        self.adversarial_loss = nn.BCELoss()

        self.teacher = teacher
        self.student = student
        self.generator = generator

        self.X = X
        self.Y = y

        self.batch_size = batch_size
        self.nb_batch = int(self.X.shape[0] / self.batch_size)

    def forward(self, weight, w_star, w_init, netD, valid):
        # self.generator.linear.weight = weight
        # self.student.lin.weight = w_init

        with torch.no_grad():
            # for param1 in self.generator.parameters():
            #    param1 = weight
            self.generator.load_state_dict(weight)
            for param1 in self.student.parameters():
                param1 = w_init
            for param2 in self.teacher.parameters():
                param2 = w_star

        loss_stu = 0
        w_loss = 0
        tau = 1

        new_weight = w_init

        model_paramters = list(self.generator.parameters())

        for i in range(self.nblock):
            w_t = self.student.lin.weight

            i = torch.randint(0, self.nb_batch, size=(1,)).item()
            i_min = i * self.batch_size
            i_max = (i + 1) * self.batch_size

            gt_x = self.X[i_min:i_max].cuda()
            y = self.Y[i_min:i_max].cuda()

            # Sample noise and labels as generator input
            # z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, gt_x.shape)))
            z = Variable(torch.randn(gt_x.shape)).cuda()

            # x = torch.cat((w_t, w_t-w_star, gt_x, y.unsqueeze(0)), dim=1)
            x = torch.cat((w_t, w_t-w_star, z), dim=1)
            generated_x = self.generator(x, y)

            # self.student.train()
            out = self.student(generated_x)

            loss = self.loss_fn(out, y.float())
            grad = torch.autograd.grad(loss, self.student.lin.weight, create_graph=True)
            # new_weight = self.student.lin.weight - 0.001 * grad[0]
            new_weight = new_weight - 0.001 * grad[0]
            self.student.lin.weight = torch.nn.Parameter(new_weight.cuda())
            # self.student.lin.weight = self.student.lin.weight - 0.001 * grad[0].cuda()

            # tau = np.exp(-i / 0.95)
            if i != -1:
                tau = 1
            else:
                tau = 0.95 * tau

            # self.student.eval()
            out_stu = self.teacher(generated_x)
            # out_stu = self.student(generated_x)
            loss_stu = loss_stu + tau * self.loss_fn(out_stu, y)

        w_loss = torch.linalg.norm(w_star - new_weight, ord=2) ** 2

        '''
        grad_stu = torch.autograd.grad(outputs=loss_stu,
                                       inputs=model_paramters,
                                       create_graph=True, retain_graph=True)



        grad_stu_w = torch.autograd.grad(outputs=w_loss,
                                       inputs=model_paramters,
                                       create_graph=True, retain_graph=True)
        '''

        # Loss measures generator's ability to fool the discriminator
        # valid = Variable(torch.cuda.FloatTensor(self.batch_size, 1).fill_(1.0), requires_grad=False)

        w_t = self.student.lin.weight

        i = torch.randint(0, self.nb_batch, size=(1,)).item()
        i_min = i * self.batch_size
        i_max = (i + 1) * self.batch_size

        gt_x = self.X[i_min:i_max].cuda()
        generated_labels = self.Y[i_min:i_max].cuda()

        # z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, gt_x.shape)))
        z = Variable(torch.randn(gt_x.shape)).cuda()

        # x = torch.cat((w_t, w_t-w_star, gt_x, generated_labels.unsqueeze(0)), dim=1)
        x = torch.cat((w_t, w_t-w_star, z), dim=1)
        generated_samples = self.generator(x, generated_labels)

        # generated_labels = generated_labels.float()

        validity = netD(generated_samples, Variable(generated_labels.type(torch.cuda.LongTensor)))
        g_loss = self.adversarial_loss(validity, valid)

        loss_stu = loss_stu + w_loss + g_loss

        grad_stu = torch.autograd.grad(outputs=loss_stu,
                                       inputs=model_paramters,
                                       create_graph=True, retain_graph=True)

        return grad_stu, loss_stu, generated_samples, generated_labels, g_loss
