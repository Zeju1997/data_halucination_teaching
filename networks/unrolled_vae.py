# https://github.com/jakeoung/BayesianUnrolling/blob/master/unroll/model/vanilla.py

# from . import attention
# from . import softargmax

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

import numpy as np

from torch import distributions as D

class Generator_old_mnist(nn.Module):
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


class Discriminator_old_mnist(nn.Module):
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

class Generator(nn.Module):
    def __init__(self, opt, teacher, student):
        super(Generator, self).__init__()

        self.opt = opt
        self.x_dims = 2
        self.z_dims = 16
        self.y_dims = 2
        self.px_sigma = 0.08

        self.label_embedding = nn.Embedding(self.opt.n_classes, self.opt.label_dim)

        in_channels = teacher.lin.weight.size(1) + student.lin.weight.size(1) + self.opt.dim + self.opt.label_dim
        # in_channels = teacher.lin.weight.size(1) + student.lin.weight.size(1) + self.opt.label_dim

        # Layers for q(z|x,y):
        self.qz_fc = nn.Sequential(
                    nn.Linear(in_features=in_channels, out_features=128),
                    nn.ReLU(),
                    nn.Linear(128, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU()
                )

        self.qz_mu = nn.Linear(in_features=128, out_features=self.z_dims)
        self.qz_pre_sp = nn.Linear(in_features=128, out_features=self.z_dims)

    def forward(self, z, y):
        h = torch.cat((z, self.label_embedding(y.to(torch.int64))), dim=1)
        # h = torch.cat((x, self.label_embedding(y.to(torch.int64))), dim=1)

        h1 = self.qz_fc(h)
        z_mu = self.qz_mu(h1)
        z_pre_sp = self.qz_pre_sp(h1)
        z_std = F.softplus(z_pre_sp)
        return self.reparameterize(z_mu, z_std), z_mu, z_std

    def reparameterize(self, mu, std):
        eps = torch.randn(mu.size()).cuda()
        # eps = eps.cuda()

        return mu + eps * std


class Generator_moon_1(nn.Module):
    def __init__(self, opt, teacher, student):
        super(Generator_moon, self).__init__()

        self.opt = opt
        self.z_dims = 20
        self.label_embedding = nn.Embedding(self.opt.n_classes, self.opt.label_dim)

        in_channels = teacher.lin.weight.size(1) + student.lin.weight.size(1) + self.opt.dim + self.opt.label_dim

        self.model = nn.Sequential(
                        nn.Linear(in_channels, self.opt.hidden_dim*4, bias=False),
                        nn.ReLU(),
                        nn.Linear(self.opt.hidden_dim*4, self.opt.hidden_dim*2, bias=False),
                        nn.ReLU(),
                    )

        self.qz_mu = nn.Linear(in_features=self.opt.hidden_dim*2, out_features=self.z_dims)
        self.qz_pre_sp = nn.Linear(in_features=self.opt.hidden_dim*2, out_features=self.z_dims)

    def forward(self, z, label):
        x = torch.cat((z, self.label_embedding(label.to(torch.int64))), dim=1)

        x = self.model(x)
        z_mu = self.qz_mu(x)
        z_pre_sp = self.qz_pre_sp(x)
        z_std = F.softplus(z_pre_sp)

        return self.reparameterize(z_mu, z_std), z_mu, z_std

    def reparameterize(self, mu, std):
        eps = torch.randn(mu.size())
        eps = eps.cuda()

        return mu + eps * std


class Generator_moon(nn.Module):
    def __init__(self, opt, teacher, student):
        super(Generator_moon, self).__init__()

        self.opt = opt
        self.x_dims = 2
        self.z_dims = 20
        self.y_dims = 2
        self.px_sigma = 0.08

        self.label_embedding = nn.Embedding(self.opt.n_classes, self.opt.label_dim)

        in_channels = teacher.lin.weight.size(1) + student.lin.weight.size(1) + self.opt.dim + self.opt.label_dim

        # Layers for q(z|x,y):
        self.qz_fc = nn.Sequential(
                    nn.Linear(in_features=in_channels, out_features=128),
                    nn.ReLU(),
                    nn.Linear(128, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU()
                )

        self.qz_mu = nn.Linear(in_features=128, out_features=self.z_dims)
        self.qz_pre_sp = nn.Linear(in_features=128, out_features=self.z_dims)

    def forward(self, z, y):
        h = torch.cat((z, self.label_embedding(y.to(torch.int64))), dim=1)
        # h = torch.cat((x, self.label_embedding(y.to(torch.int64))), dim=1)

        h1 = self.qz_fc(h)
        z_mu = self.qz_mu(h1)
        z_pre_sp = self.qz_pre_sp(h1)
        z_std = F.softplus(z_pre_sp)
        return self.reparameterize(z_mu, z_std), z_mu, z_std

    def reparameterize(self, mu, std):
        eps = torch.randn(mu.size()).cuda()
        # eps = eps.cuda()

        return mu + eps * std


class UnrolledOptimizer(nn.Module):
    """
    Args:
        - nscale : number of scales
        - alpha : scale factor in the softmax in the expansion (rho in the paper)
        - nblock : number of stages (K in the paper)
        - K : kernel size
    """
    def __init__(self, opt, teacher, student, generator, vae, X, Y, proj_matrix):
        super(UnrolledOptimizer, self).__init__()

        self.opt = opt

        self.optim_blocks = nn.ModuleList()

        self.loss_fn = nn.MSELoss()
        self.adversarial_loss = nn.BCELoss()

        self.teacher = teacher
        self.student = student
        self.generator = generator
        self.vae = vae

        self.X = X
        self.Y = Y

        self.nb_batch = int(self.X.shape[0] / self.opt.batch_size)

        self.proj_matrix = proj_matrix

    def data_sampler(self, i):
        i_min = i * self.opt.batch_size
        i_max = (i + 1) * self.opt.batch_size

        x = self.X[i_min:i_max].cuda()
        y = self.Y[i_min:i_max].cuda()

        # y = y.to(torch.int64)

        return x, y

    def forward(self, weight, w_star):
        # self.generator.linear.weight = weight
        # self.student.lin.weight = w_init

        # convert labels to onehot encoding
        cls = torch.arange(self.opt.n_classes)
        onehot = torch.zeros(self.opt.n_classes, self.opt.n_classes).scatter_(1, cls.view(self.opt.n_classes, 1), 1)
        # reshape labels to image size, with number of labels as channel
        fill = torch.zeros([self.opt.n_classes, self.opt.n_classes, self.opt.img_size, self.opt.img_size])
        for i in range(self.opt.n_classes):
            fill[i, i, :, :] = 1

        with torch.no_grad():
            # for param1 in self.generator.parameters():
            #    param1 = weight
            self.generator.load_state_dict(weight)
            self.teacher.load_state_dict(torch.load('teacher_wstar.pth'))
            self.student.load_state_dict(torch.load('teacher_w0.pth'))
            self.vae.load_state_dict(torch.load('pretrained_vae.pth'))

        loss_stu = 0
        w_loss = 0
        tau = 1

        new_weight = self.student.lin.weight

        n = 0

        model_paramters = list(self.generator.parameters())

        for i in range(self.opt.n_unroll_blocks):
            w_t = self.student.lin.weight
            w_t = w_t / torch.norm(w_t)

            i = torch.randint(0, self.nb_batch, size=(1,)).item()
            gt_x, gt_y = self.data_sampler(i)
            gt_y_onehot = onehot[gt_y.long()].cuda()
            gt_x = gt_x / torch.norm(gt_x)

            # Sample noise and labels as generator input
            # z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, gt_x.shape)))
            z = Variable(torch.randn((self.opt.batch_size, self.opt.latent_dim))).cuda()

            w = torch.cat((w_t, w_t-w_star), dim=1)
            w = w.repeat(self.opt.batch_size, 1)
            x = torch.cat((w, z), dim=1)

            z, qz_mu, qz_std = self.generator(x, gt_y)
            generated_x, y_logit = self.vae.p_xy(z)
            generated_x = generated_x.view(self.opt.batch_size, -1)
            generated_x_proj = generated_x @ self.proj_matrix.cuda()

            # self.student.train()
            out = self.student(generated_x_proj)

            loss = self.loss_fn(out, gt_y.float())

            grad = torch.autograd.grad(loss,
                                       self.student.lin.weight,
                                       create_graph=True, retain_graph=True)

            # new_weight = self.student.lin.weight - 0.001 * grad[0]
            new_weight = new_weight - 0.001 * grad[0]
            self.student.lin.weight = torch.nn.Parameter(new_weight.cuda())
            # self.student.lin.weight = self.student.lin.weight - 0.001 * grad[0].cuda()

            # tau = np.exp(-i / 0.95)
            '''
            if i != -1:
                tau = 1
            else:
                tau = 0.95 * tau
            '''

            # self.student.eval()
            out_stu = self.teacher(generated_x_proj)
            # out_stu = self.student(generated_x)
            loss_stu = loss_stu + self.loss_fn(out_stu, gt_y)

        w_loss = torch.linalg.norm(self.teacher.lin.weight - new_weight, ord=2) ** 2

        w_t = self.student.lin.weight
        w_t = w_t / torch.norm(w_t)

        i = torch.randint(0, self.nb_batch, size=(1,)).item()
        gt_x, generated_labels = self.data_sampler(i)

        # z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, gt_x.shape)))
        z = Variable(torch.randn((self.opt.batch_size, self.opt.latent_dim))).cuda()
        w = torch.cat((w_t, w_t-w_star), dim=1)
        w = w.repeat(self.opt.batch_size, 1)
        x = torch.cat((w, z), dim=1)

        z, qz_mu, qz_std = self.generator(x, gt_y)

        generated_x, y_logit = self.vae.p_xy(z)

        qz = D.normal.Normal(qz_mu, qz_std)
        qz = D.independent.Independent(qz, 1)
        pz = D.normal.Normal(torch.zeros_like(z), torch.ones_like(z))
        pz = D.independent.Independent(pz, 1)

        # For: - KL[qz || pz]
        kl_loss = D.kl.kl_divergence(qz, pz)

        loss_stu = loss_stu + w_loss + kl_loss

        grad_stu = torch.autograd.grad(outputs=loss_stu,
                                       inputs=model_paramters,
                                       create_graph=False, retain_graph=False)

        return grad_stu, loss_stu


class UnrolledOptimizer_moon(nn.Module):
    """
    Args:
        - nscale : number of scales
        - alpha : scale factor in the softmax in the expansion (rho in the paper)
        - nblock : number of stages (K in the paper)
        - K : kernel size
    """
    def __init__(self, opt, teacher, student, generator, vae, X, Y):
        super(UnrolledOptimizer_moon, self).__init__()

        self.opt = opt

        self.optim_blocks = nn.ModuleList()

        self.loss_fn = nn.MSELoss()
        self.adversarial_loss = nn.BCELoss()

        self.teacher = teacher
        self.student = student
        self.generator = generator

        self.vae = vae

        self.X = X
        self.Y = Y

        self.nb_batch = int(self.X.shape[0] / self.opt.batch_size)

    def data_sampler(self, i):
        i_min = i * self.opt.batch_size
        i_max = (i + 1) * self.opt.batch_size

        x = self.X[i_min:i_max].cuda()
        y = self.Y[i_min:i_max].cuda()

        return x, y

    def forward(self, weight, w_star):
        # self.generator.linear.weight = weight
        # self.student.lin.weight = w_init

        with torch.no_grad():
            # for param1 in self.generator.parameters():
            #    param1 = weight
            self.generator.load_state_dict(weight)
            self.student.load_state_dict(torch.load('teacher_w0.pth'))
            self.teacher.load_state_dict(torch.load('teacher_wstar.pth'))
            self.vae.load_state_dict(torch.load('pretrained_vae.pth'))
            # for param1 in self.student.parameters():
            #     param1 = w_init
            # for param2 in self.teacher.parameters():
            #     param2 = w_star

        loss_stu = 0
        w_loss = 0
        tau = 1

        new_weight = self.student.lin.weight

        model_paramters = list(self.generator.parameters())
        optimizer = torch.optim.Adam(params=self.generator.parameters(), lr=1e-3, weight_decay=1e-5)

        for i in range(self.opt.n_unroll_blocks):
            w_t = self.student.lin.weight
            w_t = w_t / torch.norm(w_t)

            i = torch.randint(0, self.nb_batch, size=(1,)).item()
            gt_x, gt_y = self.data_sampler(i)

            # Sample noise and labels as generator input
            noise = Variable(torch.randn(gt_x.shape)).cuda()
            w = torch.cat((w_t, w_t-w_star, noise), dim=1)

            z, qz_mu, qz_std = self.generator(w, gt_y)
            generated_x, x_mu, x_std, y_logit = self.vae.p_xy(z)

            # self.student.train()
            out = self.student(generated_x)

            loss = self.loss_fn(out, gt_y.float())
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
            loss_stu = loss_stu + tau * self.loss_fn(out_stu, gt_y)

        w_loss = torch.linalg.norm(self.teacher.lin.weight - new_weight, ord=2) ** 2

        w_t = self.student.lin.weight
        w_t = w_t / torch.norm(w_t)

        i = torch.randint(0, self.nb_batch, size=(1,)).item()
        gt_x, generated_labels = self.data_sampler(i)

        # z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, gt_x.shape)))
        z = Variable(torch.randn(gt_x.shape)).cuda()

        # x = torch.cat((w_t, w_t-w_star, gt_x, generated_labels.unsqueeze(0)), dim=1)
        x = torch.cat((w_t, w_t-w_star, z), dim=1)
        # generated_samples = self.generator(x, generated_labels)

        z, qz_mu, qz_std = self.generator(x, gt_y)
        # x = self.generator(w, gt_y)

        generated_x, x_mu, x_std, y_logit = self.vae.p_xy(z)

        qz = D.normal.Normal(qz_mu, qz_std)
        qz = D.independent.Independent(qz, 1)
        pz = D.normal.Normal(torch.zeros_like(z), torch.ones_like(z))
        pz = D.independent.Independent(pz, 1)

        # For: - KL[qz || pz]
        kl_loss = D.kl.kl_divergence(qz, pz)

        loss_stu = loss_stu + w_loss + kl_loss

        grad_stu = torch.autograd.grad(outputs=loss_stu,
                                       inputs=model_paramters,
                                       create_graph=True, retain_graph=True)

        return grad_stu, loss_stu
