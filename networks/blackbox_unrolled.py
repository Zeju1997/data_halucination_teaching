# https://github.com/jakeoung/BayesianUnrolling/blob/master/unroll/model/vanilla.py

# from . import attention
# from . import softargmax

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

import numpy as np

from tqdm import tqdm


def mixup_data(gt_x, generated_x, gt_y, generated_y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        # lam = 1
    else:
        lam = 1

    '''
    batch_size = gt_x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    '''
    y_a = gt_y
    y_b = generated_y
    mixed_x = lam * gt_x + (1 - lam) * generated_x

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    loss = lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    return loss.to(torch.float32)


class Generator1(nn.Module):
    def __init__(self, opt, teacher, student):
        super(Generator, self).__init__()

        self.opt = opt
        self.label_emb = nn.Embedding(self.opt.n_classes, self.opt.label_dim)
        self.img_shape = (self.opt.channels, self.opt.img_size, self.opt.img_size)

        in_channels = student.lin.weight.size(1) + self.opt.latent_dim + self.opt.label_dim

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


class Discriminator1(nn.Module):
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
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        # d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        d_in = torch.cat((img, self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity


class Generator(nn.Module):
    """ G(z) """
    def __init__(self, opt, teacher, student):
        # initalize super module
        super(Generator, self).__init__()

        self.opt = opt
        # self.label_emb = nn.Embedding(self.opt.n_classes, self.opt.label_dim)
        self.img_shape = (self.opt.channels, self.opt.img_size, self.opt.img_size)

        # in_channels = teacher.lin.weight.size(1) + student.lin.weight.size(1) + self.opt.latent_dim # + self.opt.label_dim
        in_channels = teacher.lin.weight.size(1) + self.opt.latent_dim

        # noise z input layer : (batch_size, 100, 1, 1)
        self.layer_x = nn.Sequential(nn.ConvTranspose2d(in_channels=in_channels, out_channels=128, kernel_size=3, stride=1, padding=0, bias=False),
                                                        # out size : (batch_size, 128, 3, 3)
                                                        nn.BatchNorm2d(128),
                                                        # out size : (batch_size, 128, 3, 3)
                                                        nn.ReLU(),
                                                        # out size : (batch_size, 128, 3, 3)
                                                        )

        # label input layer : (batch_size, 10, 1, 1)
        self.layer_y = nn.Sequential(nn.ConvTranspose2d(in_channels=self.opt.n_classes, out_channels=128, kernel_size=3, stride=1, padding=0, bias=False),
                                                        # out size : (batch_size, 128, 3, 3)
                                                        nn.BatchNorm2d(128),
                                                        # out size : (batch_size, 128, 3, 3)
                                                        nn.ReLU(),
                                                        # out size : (batch_size, 128, 3, 3)
                                                        )

        # noise z and label concat input layer : (batch_size, 256, 3, 3)
        self.layer_xy = nn.Sequential(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=0, bias=False),
                                                        # out size : (batch_size, 128, 7, 7)
                                                        nn.BatchNorm2d(128),
                                                        # out size : (batch_size, 128, 7, 7)
                                                        nn.ReLU(),
                                                        # out size : (batch_size, 128, 7, 7)
                                                        nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
                                                        # out size : (batch_size, 64, 14, 14)
                                                        nn.BatchNorm2d(64),
                                                        # out size : (batch_size, 64, 14, 14)
                                                        nn.ReLU(),
                                                        # out size : (batch_size, 64, 14, 14)
                                                        nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False),
                                                        # out size : (batch_size, 1, 28, 28)
                                                        nn.Tanh())
                                                        # out size : (batch_size, 1, 28, 28)

    def forward(self, x, y):
        # x size : (batch_size, 100)
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        # x size : (batch_size, 100, 1, 1)
        x = self.layer_x(x)
        # x size : (batch_size, 128, 3, 3)

        # y size : (batch_size, 10)
        y = y.view(y.shape[0], y.shape[1], 1, 1)
        # y size : (batch_size, 100, 1, 1)
        y = self.layer_y(y)
        # y size : (batch_size, 128, 3, 3)

        # concat x and y
        xy = torch.cat([x, y], dim=1)
        # xy size : (batch_size, 256, 3, 3)
        xy = self.layer_xy(xy)
        # xy size : (batch_size, 1, 28, 28)
        return xy


class Discriminator(nn.Module):
    """ D(x) """
    def __init__(self, opt):
        # initalize super module
        super(Discriminator, self).__init__()

        self.opt = opt

        # creating layer for image input , input size : (batch_size, 1, 28, 28)
        self.layer_x = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),
                                    # out size : (batch_size, 32, 14, 14)
                                    nn.LeakyReLU(0.2, inplace=True),
                                    # out size : (batch_size, 32, 14, 14)
                                    )

        # creating layer for label input, input size : (batch_size, 10, 28, 28)
        self.layer_y = nn.Sequential(nn.Conv2d(in_channels=self.opt.n_classes, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),
                                     # out size : (batch_size, 32, 14, 14)
                                     nn.LeakyReLU(0.2, inplace=True),
                                     # out size : (batch_size, 32, 14, 14)
                                     )

        # layer for concat of image layer and label layer, input size : (batch_size, 64, 14, 14)
        self.layer_xy = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
                                       # out size : (batch_size, 128, 7, 7)
                                       nn.BatchNorm2d(128),
                                       # out size : (batch_size, 128, 7, 7)
                                       nn.LeakyReLU(0.2, inplace=True),
                                       # out size : (batch_size, 128, 7, 7)
                                       nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=0, bias=False),
                                       # out size : (batch_size, 256, 3, 3)
                                       nn.BatchNorm2d(256),
                                       # out size : (batch_size, 256, 3, 3)
                                       nn.LeakyReLU(0.2, inplace=True),
                                       # out size : (batch_size, 256, 3, 3)
                                       # Notice in below layer, we are using out channels as 1, we don't need to use Linear layer
                                       # Same is recommended in DCGAN paper also
                                       nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False),
                                       # out size : (batch_size, 1, 1, 1)
                                       # sigmoid layer to convert in [0,1] range
                                       nn.Sigmoid()
                                       )

    def forward(self, x, y):
        # size of x : (batch_size, 1, 28, 28)
        x = self.layer_x(x)
        # size of x : (batch_size, 32, 14, 14)

        # size of y : (batch_size, 10, 28, 28)
        y = self.layer_y(y)
        # size of y : (batch_size, 32, 14, 14)

        # concat image layer and label layer output
        xy = torch.cat([x,y], dim=1)
        # size of xy : (batch_size, 64, 14, 14)
        xy = self.layer_xy(xy)
        # size of xy : (batch_size, 1, 1, 1)
        xy = xy.view(xy.shape[0], -1)
        # size of xy : (batch_size, 1)
        return xy


class Generator_moon(nn.Module):
    def __init__(self, opt, teacher, student):
        super(Generator_moon, self).__init__()

        self.opt = opt
        self.label_embedding = nn.Embedding(self.opt.n_classes, self.opt.label_dim)

        # in_channels = teacher.lin.weight.size(1) + student.lin.weight.size(1) + self.opt.dim + self.opt.label_dim
        in_channels = teacher.lin.weight.size(1) + self.opt.dim + self.opt.label_dim

        self.input_fc = nn.Linear(in_channels, self.opt.hidden_dim*4, bias=False)
        self.hidden_fc = nn.Linear(self.opt.hidden_dim*4, self.opt.hidden_dim*2, bias=False)
        self.output_fc = nn.Linear(self.opt.hidden_dim*2, self.opt.dim, bias=False)
        # self.activation = nn.LeakyReLU(0.1)
        self.activation = nn.ReLU()
        self.out_activation = nn.Sigmoid()

    def forward(self, z, label):
        x = torch.cat((z, self.label_embedding(label.to(torch.int64))), dim=1)
        x = self.activation(self.input_fc(x))
        x = self.activation(self.hidden_fc(x))
        x = self.output_fc(x)
        # x = self.out_activation(x) * 4 - 2
        return x


class Discriminator_moon(nn.Module):
    def __init__(self, opt):
        super(Discriminator_moon, self).__init__()

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


class UnrolledBlackBoxOptimizer_moon(nn.Module):
    """
    Args:
        - nscale : number of scales
        - alpha : scale factor in the softmax in the expansion (rho in the paper)
        - nblock : number of stages (K in the paper)
        - K : kernel size
    """
    def __init__(self, opt, teacher, student, generator, X, Y, proj_matrix=None):
        super(UnrolledBlackBoxOptimizer_moon, self).__init__()

        self.opt = opt

        self.optim_blocks = nn.ModuleList()

        self.loss_fn = nn.MSELoss()
        # self.adversarial_loss = nn.BCELoss()
        self.adversarial_loss = nn.MSELoss()

        self.teacher = teacher
        self.student = student
        self.generator = generator

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

    def forward(self, weight, w_star, w_init, netD=None, valid=None):
        # self.generator.linear.weight = weight
        # self.student.lin.weight = w_init

        with torch.no_grad():
            # for param1 in self.generator.parameters():
            #    param1 = weight
            # self.generator.load_state_dict(weight)
            # for param1 in self.student.parameters():
            #     param1 = w_init
            # for param2 in self.teacher.parameters():
            #     param2 = w_star
            self.generator.load_state_dict(weight)
            self.teacher.load_state_dict(torch.load('teacher_wstar.pth'))
            self.student.load_state_dict(torch.load('teacher_w0.pth'))

        loss_stu = 0
        w_loss = 0
        tau = 1

        new_weight = self.student.lin.weight

        model_paramters = list(self.generator.parameters())

        for i in range(self.opt.n_unroll_blocks):
            w_t = self.student.lin.weight
            w_t = w_t / torch.norm(w_t)

            i = torch.randint(0, self.nb_batch, size=(1,)).item()
            gt_x, gt_y = self.data_sampler(i)

            # Sample noise and labels as generator input
            # z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, gt_x.shape)))
            z = Variable(torch.randn((self.opt.batch_size, self.opt.latent_dim))).cuda()

            # x = torch.cat((w_t, w_t-w_star, gt_x, y.unsqueeze(0)), dim=1)
            x = torch.cat((w_t, z), dim=1)
            generated_x = self.generator(x, gt_y)

            # generated_x = generated_x @ self.proj_matrix.cuda()

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
            # out_stu = self.student(gt_x)
            # loss_stu = loss_stu + tau * self.loss_fn(out_stu, gt_y)

        out_stu = new_weight @ torch.transpose(gt_x, 0, 1)
        loss_stu = loss_stu + tau * self.loss_fn(out_stu, gt_y)

        # w_loss = torch.linalg.norm(w_star - new_weight, ord=2) ** 2

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
        x = torch.cat((w_t, z), dim=1)
        generated_samples = self.generator(x, generated_labels)

        # generated_labels = generated_labels.float()

        validity = netD(generated_samples, Variable(generated_labels.type(torch.cuda.LongTensor)))
        g_loss = self.adversarial_loss(validity, valid)

        # loss_stu = loss_stu + w_loss + g_loss

        grad_stu = torch.autograd.grad(outputs=loss_stu,
                                       inputs=model_paramters,
                                       create_graph=True, retain_graph=True)

        return grad_stu, loss_stu, generated_samples, generated_labels, g_loss


class UnrolledBlackBoxOptimizer(nn.Module):
    """
    Args:
        - nscale : number of scales
        - alpha : scale factor in the softmax in the expansion (rho in the paper)
        - nblock : number of stages (K in the paper)
        - K : kernel size
    """
    def __init__(self, opt, teacher, student, generator, X, Y, proj_matrix=None):
        super(UnrolledBlackBoxOptimizer, self).__init__()

        self.opt = opt
        self.optim_blocks = nn.ModuleList()

        self.loss_fn = nn.MSELoss()
        self.adversarial_loss = nn.BCELoss()

        self.teacher = teacher
        self.student = student
        self.generator = generator

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

    def forward(self, weight, w_star, w_init, netD, generated_labels, real, proj_matrix):
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
            # for param1 in self.student.parameters():
            #     param1 = w_init
            # for param2 in self.teacher.parameters():
            #     param2 = w_star

        loss_stu = 0
        w_loss = 0
        tau = 1

        new_weight = self.student.lin.weight

        w_init = self.student.lin.weight
        w_init = w_init / torch.norm(w_init)

        model_paramters = list(self.generator.parameters())

        train_loss = []

        for i in range(self.opt.n_unroll_blocks):
            w_t = self.student.lin.weight
            w_t = w_t / torch.norm(w_t)

            i = torch.randint(0, self.nb_batch, size=(1,)).item()
            gt_x, gt_y = self.data_sampler(i)
            gt_y_onehot = onehot[gt_y.long()].cuda()

            # Sample noise and labels as generator input
            # z = Variable(torch.randn(gt_x.shape)).cuda()
            z = Variable(torch.randn((self.opt.batch_size, self.opt.latent_dim))).cuda()

            # gt_x = gt_x.reshape(self.opt.batch_size, -1)
            # z = gt_x @ proj_matrix.cuda()

            gt_x_norm = gt_x / torch.norm(gt_x)

            # x = torch.cat((w_t, w_t-w_star, z), dim=1)
            # w = torch.cat((w_t, w_t-w_init), dim=1)
            w = w_t.repeat(self.opt.batch_size, 1)
            x = torch.cat((w, gt_x_norm), dim=1)
            generated_x = self.generator(x, gt_y_onehot)

            generated_x = generated_x.view(self.opt.batch_size, -1)
            generated_x = generated_x @ self.proj_matrix.cuda()

            # self.student.train()
            # out = self.student(mixed_x)
            out = self.student(generated_x)

            loss = self.loss_fn(out, gt_y.float())

            grad = torch.autograd.grad(loss,
                                       self.student.lin.weight,
                                       create_graph=True, retain_graph=True)

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
            # out_stu = self.teacher(generated_x)
            # out_stu = self.student(generated_x)
            # out_stu = self.student(gt_x)

            # loss_stu = loss_stu + tau * self.loss_fn(out_stu, gt_y)

        out_stu = new_weight @ torch.transpose(gt_x, 0, 1)
        loss_stu = loss_stu + tau * self.loss_fn(out_stu, gt_y)

        # print(n, "After backward pass", torch.cuda.memory_allocated(0))

        # train_loss.append(loss.item())

        # out_stu = new_weight @ torch.transpose(gt_x, 0, 1)
        # loss_stu = self.loss_fn(out_stu, gt_y)

        # w_loss = torch.linalg.norm(self.teacher.lin.weight - new_weight, ord=2) ** 2

        w_t = self.student.lin.weight
        w_t = w_t / torch.norm(w_t)
        z = torch.randn(self.opt.batch_size, self.opt.latent_dim).cuda()

        # w = torch.cat((w_t, w_t-w_init), dim=1)
        w = w_t.repeat(self.opt.batch_size, 1)
        # w = w_t.repeat(self.opt.batch_size, 1)
        gt_x_norm = gt_x / torch.norm(gt_x)
        x = torch.cat((w, gt_x_norm), dim=1)

        # generated_labels = (torch.rand(self.opt.batch_size, 1)*2).type(torch.LongTensor).squeeze(1)
        generated_labels_onehot = onehot[generated_labels].cuda()
        generated_labels_fill = fill[generated_labels].cuda()

        generated_samples = self.generator(x, generated_labels_onehot)

        z_out = netD(generated_samples, generated_labels_fill)
        g_loss = self.adversarial_loss(z_out, real)

        alpha = 1 # 0.001
        # loss_stu = loss_stu / (self.opt.n_unroll_blocks * alpha)
        loss_final = loss_stu
        # loss_final = loss_stu + alpha * g_loss
        # loss_final = g_loss

        # ratio = alpha * g_loss.item() / loss_final.item()
        # print("ratio", ratio)

        grad_stu = torch.autograd.grad(outputs=loss_final,
                                       inputs=model_paramters,
                                       create_graph=False, retain_graph=False)

        return grad_stu, loss_stu, g_loss, z_out, generated_samples, train_loss
