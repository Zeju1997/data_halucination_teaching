# https://github.com/jakeoung/BayesianUnrolling/blob/master/unroll/model/vanilla.py

# from . import attention
# from . import softargmax

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P

from torch.autograd import Variable

import numpy as np
import torch.optim as optim

import matplotlib.pyplot as plt
from torchvision.utils import save_image

from tqdm import tqdm

import os

import copy

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


class Generator(nn.Module):
    def __init__(self, opt, teacher, student):
        super(Generator, self).__init__()

        self.opt = opt
        self.label_emb = nn.Embedding(self.opt.n_classes, self.opt.label_dim)
        self.img_shape = (self.opt.channels, self.opt.img_size, self.opt.img_size)

        # in_channels = student.lin.weight.size(1) + self.opt.latent_dim + self.opt.label_dim
        in_channels = 512 + self.opt.label_dim

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
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        # d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        d_in = torch.cat((img, self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity


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


class CloseToMask(nn.Module):
    def __init__(self, mask, S, eps=0.1, f=torch.tanh):
        super().__init__()
        self.register_buffer("mask", mask)
        self.register_buffer("S_mask", S[mask])
        self.eps = eps
        self.f = f

    def forward(self, X):
        Y = self.S_mask + self.eps * self.f(X[self.mask])
        Z = X.masked_scatter(self.mask, Y)
        return Z


def project_onto_l1_ball1(x_orig, x, eps):
    """
    Compute Euclidean projection onto the L1 ball for a batch.

      min ||x - u||_2 s.t. ||u||_1 <= eps

    Inspired by the corresponding numpy version by Adrien Gaidon.

    Parameters
    ----------
    x: (batch_size, *) torch array
      batch of arbitrary-size tensors to project, possibly on GPU

    eps: float
      radius of l-1 ball to project onto

    Returns
    -------
    u: (batch_size, *) torch array
      batch of projected tensors, reshaped to match the original

    Notes
    -----
    The complexity of this algorithm is in O(dlogd) as it involves sorting x.

    References
    ----------
    [1] Efficient Projections onto the l1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
    """
    # original_shape = x.shape
    # x = x.view(x.shape[0], -1)
    diff = torch.norm(x - x_orig, p=1, dim=1)
    mask = (diff < eps).float().unsqueeze(1)
    mu, _ = torch.sort(torch.abs(x), dim=1, descending=True)
    cumsum = torch.cumsum(mu, dim=1)
    arange = torch.arange(1, x.shape[1] + 1, device=x.device)
    rho, _ = torch.max((mu * arange > (cumsum - eps)) * arange, dim=1)
    theta = (cumsum[torch.arange(x.shape[0]), rho.cpu() - 1] - eps) / rho
    proj = (torch.abs(x) - theta.unsqueeze(1)).clamp(min=0)
    x_proj = mask * x + (1 - mask) * proj * torch.sign(x) + x_orig

    diff = torch.norm(x_proj - x_orig, p=1, dim=1)
    mask = (diff < eps).float().unsqueeze(1)
    return x_proj


def project_onto_l1_ball(x, eps):
    """
    Compute Euclidean projection onto the L1 ball for a batch.

      min ||x - u||_2 s.t. ||u||_1 <= eps

    Inspired by the corresponding numpy version by Adrien Gaidon.

    Parameters
    ----------
    x: (batch_size, *) torch array
      batch of arbitrary-size tensors to project, possibly on GPU

    eps: float
      radius of l-1 ball to project onto

    Returns
    -------
    u: (batch_size, *) torch array
      batch of projected tensors, reshaped to match the original

    Notes
    -----
    The complexity of this algorithm is in O(dlogd) as it involves sorting x.

    References
    ----------
    [1] Efficient Projections onto the l1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
    """
    original_shape = x.shape
    x = x.view(x.shape[0], -1)
    mask = (torch.norm(x, p=1, dim=1) < eps).float().unsqueeze(1)
    mu, _ = torch.sort(torch.abs(x), dim=1, descending=True)
    cumsum = torch.cumsum(mu, dim=1)
    arange = torch.arange(1, x.shape[1] + 1, device=x.device)
    rho, _ = torch.max((mu * arange > (cumsum - eps)) * arange, dim=1)
    theta = (cumsum[torch.arange(x.shape[0]), rho.cpu() - 1] - eps) / rho
    proj = (torch.abs(x) - theta.unsqueeze(1)).clamp(min=0)
    x_proj = mask * x + (1 - mask) * proj * torch.sign(x)

    mask = (torch.norm(x, p=1, dim=1) < eps).float().unsqueeze(1)
    return x_proj.view(original_shape)


class UnrolledBlackBoxOptimizer(nn.Module):
    """
    Args:
        - nscale : number of scales
        - alpha : scale factor in the softmax in the expansion (rho in the paper)
        - nblock : number of stages (K in the paper)
        - K : kernel size
    """
    def __init__(self, opt, loader):
        super(UnrolledBlackBoxOptimizer, self).__init__()

        self.opt = opt

        self.optim_blocks = nn.ModuleList()

        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn_w = nn.MSELoss()

        self.adversarial_loss = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()

        # self.teacher = teacher
        # self.student = student
        # self.generator = generator

        self.loader = loader
        self.data_iter = iter(loader)
        # self.Y = y

        # self.nb_batch = int(self.X.shape[0] / self.opt.batch_size)

        # self.proj_matrix = proj_matrix

    def optimize_latent_features(self, fc, z, labels):
        """Run the style transfer."""
        # print('Building the style transfer model..')

        z_org = z.detach().clone()
        z_tmp = z.detach().clone()

        z_tmp.requires_grad_(True)
        # fc.requires_grad_(False)

        optimizer = optim.SGD([z_tmp], lr=0.1, momentum=0.9)

        optim_loss = []

        # print('Optimizing..')
        run = [0]
        while run[0] <= 100:
            # correct the values of updated input image
            # with torch.no_grad():
            #    z.clamp_(0, 1)

            optimizer.zero_grad()
            output = fc(z_tmp)

            loss = self.loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            optim_loss.append(loss.item())

            run[0] += 1
            # if run[0] % 1 == 0:
            #     print("run {}:".format(run))
            #     print('Loss : {:4f}'.format(loss.item()))
            #     print()

        # a last correction...
        # with torch.no_grad():
        #    input_img.clamp_(0, 1)

        # fig = plt.figure()
        # plt.plot(optim_loss, c="b", label="Mixup")
        # plt.xlabel("Epoch")
        # plt.ylabel("Accuracy")
        # plt.legend()
        # plt.show()

        # diff = z_org - z_tmp
        # z = z - diff

        return z, z_tmp

    def get_w_star(self, model, fc, z, labels):
        """Run the style transfer."""
        # print('Building the style transfer model..')

        optim = torch.optim.SGD([{'params': model.parameters()}, {'params': fc.parameters()}], lr=self.opt.lr, momentum=0.9, weight_decay=self.opt.decay)
        optim_loss = []

        eps = 500
        eps_batch = torch.ones(self.opt.batch_size) * eps
        eps_batch = eps_batch.cuda()

        data_iter = iter(self.loader)
        for _ in range(1):
            try:
                (inputs, targets) = data_iter.next()
            except:
                data_iter = iter(self.loader)
                (inputs, targets) = data_iter.next()

            inputs, targets = inputs.cuda(), targets.long().cuda()
            optim.zero_grad()
            z_new = model(inputs)

            diff = torch.norm(z - z_new, p='fro', dim=1) ** 2
            print('diff', diff.max())
            # mask = (diff < eps).float().unsqueeze(1)

            # alpha = (diff / eps_batch) ** 2
            # alpha = alpha.unsqueeze(1)
            z_proj = z + (z - z_new) * (eps / diff)

            diff2 = torch.norm(z - z_proj, p='fro', dim=1) ** 2
            print('diff2', diff2.max())
            # x_proj = mask * x + (1 - mask) * proj * torch.sign(x)

            # offset = project_onto_l1_ball(diff, 0.1)
            # z_proj = z + offset

            output = fc(z_proj)
            loss = self.loss_fn(output, targets)
            loss.backward()
            optim.step()

            optim_loss.append(loss.item())

        # fig = plt.figure()
        # plt.plot(optim_loss, c="b", label="Mixup")
        # plt.xlabel("Epoch")
        # plt.ylabel("Accuracy")
        # plt.legend()
        # plt.show()

        # diff = z_org - z_tmp
        # z = z - diff

        return model

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

        for _ in tqdm(range(1)):
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.cuda(), targets.long().cuda()
                z0 = model(inputs)
                z = z0

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

                diff = pdist(z, z0)

                # diff = torch.norm(z - z_new, p='fro', dim=1) ** 2
                print('diff', diff.max())
                # mask = (diff < eps).float().unsqueeze(1)

                output = fc(z)
                loss = self.loss_fn(output, targets)
                loss.backward()
                optim.step()

                optim_loss.append(loss.item())

        return z

    def forward(self, model, fc):
        # self.generator.linear.weight = weight
        # self.student.lin.weight = w_init

        # with torch.no_grad():
            # for param1 in self.generator.parameters():
            #    param1 = weight
            # self.generator.load_state_dict(weight)
            # self.teacher.load_state_dict(torch.load('teacher_wstar.pth'))
            # self.student.load_state_dict(torch.load('teacher_w0.pth'))
            # for param1 in self.student.parameters():
            #     param1 = w_init
            # for param2 in self.teacher.parameters():
            #     param2 = w_star

        loss_stu = 0
        w_loss = 0
        tau = 1

        optim = torch.optim.SGD(fc.parameters(), lr=0.001)
        model.eval()
        num_steps = 10

        optim_loss = []

        for _ in range(num_steps):
            try:
                (inputs, targets) = data_iter.next()
            except:
                data_iter = iter(self.loader)
                (inputs, targets) = data_iter.next()

            inputs, targets = inputs.cuda(), targets.long().cuda()

            z = model(inputs)

            optim.zero_grad()

            outputs = fc(z)
            loss = self.loss_fn(outputs, targets)
            loss.backward()

            optim_loss.append(loss.item())

            optim.step()

        weight = fc.state_dict()

        fig = plt.figure()
        plt.plot(optim_loss, c="b", label="Teacher (CNN)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

        '''
        new_weight = self.student.lin.weight

        w_init = self.student.lin.weight
        w_init = w_init / torch.norm(w_init)

        model_paramters = list(self.generator.parameters())
        student_paramters = list(self.student.parameters())

        train_loss = []

        self.generator.train()

        for i in range(self.opt.n_unroll_blocks):
            w_t = self.student.lin.weight
            w_t = w_t / torch.norm(w_t)

            i = torch.randint(0, self.nb_batch, size=(1,)).item()
            gt_x, gt_y = self.data_sampler(i)

            # Sample noise and labels as generator input
            # z = Variable(torch.randn(gt_x.shape)).cuda()
            z = Variable(torch.randn((self.opt.batch_size, self.opt.latent_dim))).cuda()

            # x = torch.cat((w_t, w_t-w_star, z), dim=1)
            # w = torch.cat((w_t, w_t-w_init), dim=1)
            w = w_t.repeat(self.opt.batch_size, 1)
            x = torch.cat((w, gt_x), dim=1)
            # generated_x = self.generator(x, gt_y_onehot)
            generated_x = self.generator(x, gt_y)

            # self.student.train()

            if self.proj_matrix is not None:
                generated_x = generated_x @ self.proj_matrix.cuda()

            out = self.student(generated_x)

            loss = self.loss_fn(out, gt_y.unsqueeze(1).float())

            grad = torch.autograd.grad(loss,
                                       self.student.lin.weight,
                                       create_graph=True, retain_graph=True)

            new_weight = new_weight - 0.001 * grad[0]
            self.student.lin.weight = torch.nn.Parameter(new_weight.cuda())
            # self.student.lin.weight.data = self.student.lin.weight.data - 0.001 * grad[0]
            # self.student.lin.weight = self.student.lin.weight - 0.001 * grad[0].cuda()

            # with torch.no_grad():
            # for p, g in zip(self.student.parameters(), grad):
            #    p.grad = g

            # student_optim.step()

            # tau = np.exp(-i / 0.95)
            if i != -1:
                tau = 1
            else:
                tau = 0.95 * tau

        act = torch.nn.Sigmoid()
        out_stu = new_weight @ torch.transpose(gt_x, 0, 1)
        out_stu = act(out_stu)
        loss_stu = loss_stu + tau * self.loss_fn(out_stu, gt_y.unsqueeze(1).float())

        grad_stu = torch.autograd.grad(outputs=loss_stu,
                                       inputs=model_paramters,
                                       create_graph=False, retain_graph=False)
        '''
        return weight

    def forward1(self, model, fc, model_star, inputs, targets):

        # ---------------------
        #  Optimize Linear Classifier
        # ---------------------

        model_parameters = list(model.parameters()) # + list(fc.parameters())
        model_optim = torch.optim.SGD([{'params': model.parameters()}], lr=0.1, momentum=0.9, weight_decay=self.opt.decay)
        model_optim.zero_grad()

        z = model(inputs)

        # fc_mdl = copy.deepcopy(fc)
        # model_mdl = copy.deepcopy(model)

        # z, z_optimized = self.optimize_latent_features(fc, z, targets)
        # w_star = self.get_w_star(model_mdl, fc_mdl, z, targets)
        z_optimized = model_star(inputs)

        # out = fc(z)

        # diff = fc.lin.weight - w_star
        # w_loss = torch.linalg.norm(diff, ord=2) ** 2
        w_loss = self.loss_fn_w(z, z_optimized)
        # loss = self.loss_fn(out, targets)

        gradients = torch.autograd.grad(outputs=w_loss,
                                        inputs=model_parameters,
                                        create_graph=True, retain_graph=True)

        with torch.no_grad():
            for p, g in zip(model_parameters, gradients):
                p.grad = g

        model_optim.step()

        # ---------------------
        #  Reconstruct Images from Feature Space
        # ---------------------

        # loss = self.cross_entropy(out, targets)

        # targets_onehot = torch.FloatTensor(targets.shape[0], self.opt.n_classes).cuda()
        # targets_onehot.zero_()
        # targets_onehot.scatter_(1, targets.unsqueeze(1), 1)

        # generated_inputs = self.F_inverse(model=model, netG=netG, input=inputs, class_vector=targets_onehot, features_ini=z_optimized)

        # return generated_inputs
        return model.state_dict(), w_loss.item()

    def F_inverse1(self, model, netG, input, class_vector, features_ini, cov_matrix):
        truncation = 1
        lr1 = 100
        lr2 = 0.1

        epoch1 = 8 # 8000
        epoch2 = 8 # 8000

        print_freq = 500

        schedule1 = [2000, 4000, 6000, 8000]
        schedule2 = [2000, 4000, 6000, 8000]

        size = 32

        eta = 0.005
        aug_num = 1
        aug_alpha = 0.2

        feature_num = 512

        '''Step 1'''

        noise_vector = torch.nn.Parameter(torch.randn(self.opt.batch_size, 100, requires_grad=True).cuda())
        # noise_vector = noise_vector.cuda()
        noise_vector.requires_grad = True
        print('Initial noise_vector:', noise_vector.size())

        mse_loss = torch.nn.MSELoss(reduction='sum')
        opt1 = optim.Adam([{'params': noise_vector}], lr=lr1, weight_decay=1e-4)

        for epoch in range(epoch1):
            if epoch in schedule1:
                for paras in opt1.param_groups:
                    paras['lr'] /= 10
                    print(paras['lr'])

            noise_vector_normalized = (noise_vector - noise_vector.mean()).div(noise_vector.std())
            fake_img = netG(noise_vector_normalized, class_vector)

            if epoch % print_freq == 0:
                save_image(fake_img.detach(), '{}/step1_epoch_{}.jpg'.format(self.opt.log_path, epoch), nrow=16, padding=0, normalize=True)

            features_ini = features_ini.cuda()
            input = input.cuda()
            # input_interpolate = F.interpolate(input, size=(size, size), mode='bilinear', align_corners=True).cuda()

            # fake_img_224 = F.interpolate(fake_img, size=(224, 224), mode='bilinear', align_corners=True)
            # fake_img_224.require_grad = True
            fake_img_norm = (fake_img - fake_img.mean()).div(fake_img.std())
            fake_img_norm.require_grad = True
            fake_img_norm = fake_img_norm.cuda()

            feature_fake_img = model(fake_img_norm)

            loss1 = mse_loss(feature_fake_img, features_ini)
            loss2 = eta * mse_loss(fake_img_norm, input)
            loss_a = loss1 + loss2
            opt1.zero_grad()
            loss_a.backward(retain_graph=True)  # retain_graph=True
            opt1.step()


        '''Step 2'''
        # noise_vector_batch = noise_vector.expand(aug_num, 128)
        # noise_vector_batch = torch.nn.Parameter(noise_vector_batch.cuda())
        # noise_vector_batch.requires_grad = True
        # class_vector_batch = class_vector.expand(aug_num, 1000).cuda()

        # feature_origin_batch = features_ini.expand(aug_num, feature_num).float().cuda()
        # feature_objective_batch = feature_origin_batch

        noise_vector_batch = torch.nn.Parameter(noise_vector.cuda())
        noise_vector_batch.requires_grad = True
        class_vector_batch = class_vector
        feature_origin_batch = features_ini
        feature_objective_batch = feature_origin_batch


        '''save the reconstructed image'''
        # noise_vector = noise_vector.view(1, -1)
        # noise_vector_normalized = (noise_vector - noise_vector.mean()) / noise_vector.std()
        # init_fake_img = netG(noise_vector_normalized, class_vector, truncation)
        # save_image(init_fake_img.detach().cpu(), '{0}/reconstruct'.format(self.opt.log_path))

        # noise_vector = noise_vector.view(1, -1)
        noise_vector_normalized = (noise_vector - noise_vector.mean()) / noise_vector.std()
        init_fake_img = netG(noise_vector_normalized, class_vector)
        save_image(init_fake_img.detach(), '{}/reconstruct.jpg'.format(self.opt.log_path), nrow=16, padding=0, normalize=True)

        '''
        # cov_matrix = cov_matrix.cpu().numpy()
        for i in range(cov_matrix.shape[0]):
            csv_name = os.path.join(self.opt.log_path, '{0}_cov_imagenet.csv'.format(i))
            f = open(csv_name, 'w')
            for j in range(cov_matrix.shape[1]):
                item = cov_matrix[i][j]
                f.write(str(cov_matrix[i][j]) + '\n')
            f.close()

        csv_name = os.path.join(self.opt.log_path, '{0}_cov_imagenet.csv'.format(1))
        with open(csv_name, encoding='utf-8') as f:
            cov = np.loadtxt(f, delimiter="\n")
        # CV = np.diag(cov)
        CV = np.diagonal(cov_matrix.cpu(), axis1=1, axis2=2)
        print("CV:", CV.shape)

        print("====> Start Augmentating")
        for i in range(aug_num):
            aug_np = np.random.multivariate_normal([0 for ij in range(feature_num)], aug_alpha * CV)
            aug = torch.Tensor(aug_np).float().cuda()
            print("aug[{0}]:".format(i), aug.size())
            print("feature_origin_batch[i].size(): ", feature_origin_batch[i].size())
            feature_objective_batch[i] = (feature_origin_batch[i] + aug).detach()
        print("====> End Augmentating")
        '''
        mse_loss = torch.nn.MSELoss(reduction='sum')
        opt2 = optim.SGD([{'params': noise_vector_batch}], lr=lr2, momentum=0.9, weight_decay=1e-4, nesterov=True)

        for epoch in range(epoch2):
            if epoch in schedule2:
                for paras in opt2.param_groups:
                    paras['lr'] /= 10
                    print("lr:", paras['lr'])

            n_mean = noise_vector_batch.mean(axis=1).unsqueeze(1).expand(noise_vector_batch.size(0), noise_vector_batch.size(1))
            n_std = noise_vector_batch.std(axis=1).unsqueeze(1).expand(noise_vector_batch.size(0), noise_vector_batch.size(1))
            noise_vector_normalized_batch = (noise_vector_batch - n_mean) / n_std
            fake_img_batch = netG(noise_vector_normalized_batch, class_vector_batch)

            if epoch % print_freq == 0:
                save_image(init_fake_img.detach(), '{}/step2_epoch_{}.jpg'.format(self.opt.log_path, epoch), nrow=16, padding=0, normalize=True)

                # for i in range(fake_img_batch.size(0)):
                #     save_image(fake_img_batch[i].unsqueeze(0).detach().cpu(), '{0}/step2_epoch_{1}_img_{2}'.format(self.opt.log_path, epoch // print_freq, i))
                print("noise_vector_batch:", noise_vector_batch)

            # fake_img_224 = F.interpolate(fake_img_batch, size=(224, 224), mode='bilinear', align_corners=True)
            # fake_img_224.require_grad = True

            fake_img_batch.require_grad = True

            # _fake_img_224 = fake_img_224.view(fake_img_224.size(0), -1)
            # f_mean = _fake_img_224.mean(axis=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(fake_img_224.size(0), 3, 224, 224)
            # f_std = _fake_img_224.std(axis=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(fake_img_224.size(0), 3, 224, 224)
            # fake_img_norm = (fake_img_224 - f_mean) / f_std
            # fake_img_norm = fake_img_norm.cuda()
            # fake_img_norm.require_grad = True

            _fake_img = fake_img_batch.view(fake_img_batch.size(0), -1)
            f_mean = _fake_img.mean(axis=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(fake_img_batch.size(0), 3, 32, 32)
            f_std = _fake_img.std(axis=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(fake_img_batch.size(0), 3, 32, 32)
            fake_img_norm = (fake_img_batch - f_mean) / f_std
            fake_img_norm = fake_img_norm.cuda()
            fake_img_norm.require_grad = True

            feature_fake_img_batch = model(fake_img_norm)
            loss_b = mse_loss(feature_fake_img_batch, feature_objective_batch)
            opt2.zero_grad()
            loss_b.backward(retain_graph=True)
            opt2.step()

            loss_file = os.path.join(self.opt.log_path, 'loss.txt')
            if epoch % 10 == 0:
                # print('Step2: Epoch: %d  loss_b: %.5f' % (epoch, loss_b.data.item()))
                fd = open(loss_file, 'a+')
                string = ('Step2: Epoch: {0}\t'
                         'loss_b {1}\t'.format(epoch, loss_b.data.item()))
                print(string)
                fd.write(string + '\n')
                fd.close()

    def F_inverse(self, model, netG, input, class_vector, features_ini):
        lr1 = 100
        lr2 = 0.1

        epoch1 = 1000
        epoch2 = 8000

        print_freq = 500

        schedule1 = [250, 500, 750, 1000] # [2000, 4000, 6000, 8000]
        schedule2 = [2000, 4000, 6000, 8000]

        size = 32

        eta = 0.005

        feature_num = 512

        '''Step 1 - Find corresponding noise vectors for the input images'''

        noise_vector = torch.nn.Parameter(torch.randn(self.opt.batch_size, 100, requires_grad=True).cuda())
        # noise_vector = noise_vector.cuda()
        noise_vector.requires_grad = True
        # print('Initial noise_vector:', noise_vector.size())

        mse_loss = torch.nn.MSELoss(reduction='sum')
        opt1 = optim.Adam([{'params': noise_vector}], lr=lr1, weight_decay=1e-4)

        for epoch in tqdm(range(epoch1)):
            if epoch in schedule1:
                for paras in opt1.param_groups:
                    paras['lr'] /= 10
                    # print(paras['lr'])

            noise_vector_normalized = (noise_vector - noise_vector.mean()).div(noise_vector.std())
            fake_img = netG(noise_vector_normalized, class_vector)

            # if epoch % print_freq == 0:
            #     save_image(fake_img.detach(), '{}/images/step1_epoch_{}.jpg'.format(self.opt.log_path, epoch), nrow=16, padding=0, normalize=True)

            features_ini = features_ini.cuda()
            input = input.cuda()
            # input_interpolate = F.interpolate(input, size=(size, size), mode='bilinear', align_corners=True).cuda()

            # fake_img_224 = F.interpolate(fake_img, size=(224, 224), mode='bilinear', align_corners=True)
            # fake_img_224.require_grad = True
            fake_img_norm = (fake_img - fake_img.mean()).div(fake_img.std())
            fake_img_norm.require_grad = True
            fake_img_norm = fake_img_norm.cuda()

            feature_fake_img = model(fake_img_norm)

            loss1 = mse_loss(feature_fake_img, features_ini)
            loss2 = eta * mse_loss(fake_img_norm, input)
            loss_a = loss1 + loss2
            opt1.zero_grad()
            loss_a.backward(retain_graph=True)  # retain_graph=True
            opt1.step()

        fake_img_batch = netG(noise_vector, class_vector)

        '''
        # Step 2 - Find corresponding noise vectors for the input images
        # noise_vector_batch = noise_vector.expand(aug_num, 128)
        # noise_vector_batch = torch.nn.Parameter(noise_vector_batch.cuda())
        # noise_vector_batch.requires_grad = True
        # class_vector_batch = class_vector.expand(aug_num, 1000).cuda()

        # feature_origin_batch = features_ini.expand(aug_num, feature_num).float().cuda()
        # feature_objective_batch = feature_origin_batch

        noise_vector_batch = torch.nn.Parameter(noise_vector.cuda())
        noise_vector_batch.requires_grad = True
        class_vector_batch = class_vector
        feature_objective_batch = features_ini

        # save the reconstructed image
        # noise_vector = noise_vector.view(1, -1)
        # noise_vector_normalized = (noise_vector - noise_vector.mean()) / noise_vector.std()
        # init_fake_img = netG(noise_vector_normalized, class_vector, truncation)
        # save_image(init_fake_img.detach().cpu(), '{0}/reconstruct'.format(self.opt.log_path))

        # noise_vector = noise_vector.view(1, -1)
        noise_vector_normalized = (noise_vector - noise_vector.mean()) / noise_vector.std()
        init_fake_img = netG(noise_vector_normalized, class_vector)
        save_image(init_fake_img.detach(), '{}/reconstruct.jpg'.format(self.opt.log_path), nrow=16, padding=0, normalize=True)

        mse_loss = torch.nn.MSELoss(reduction='sum')
        opt2 = optim.SGD([{'params': noise_vector_batch}], lr=lr2, momentum=0.9, weight_decay=1e-4, nesterov=True)

        for epoch in tqdm(range(epoch2)):
            if epoch in schedule2:
                for paras in opt2.param_groups:
                    paras['lr'] /= 10
                    # print("lr:", paras['lr'])

            n_mean = noise_vector_batch.mean(axis=1).unsqueeze(1).expand(noise_vector_batch.size(0), noise_vector_batch.size(1))
            n_std = noise_vector_batch.std(axis=1).unsqueeze(1).expand(noise_vector_batch.size(0), noise_vector_batch.size(1))
            noise_vector_normalized_batch = (noise_vector_batch - n_mean) / n_std
            fake_img_batch = netG(noise_vector_normalized_batch, class_vector_batch)

            # if epoch % print_freq == 0:
            #     save_image(init_fake_img.detach(), '{}/images/step2_epoch_{}.jpg'.format(self.opt.log_path, epoch), nrow=16, padding=0, normalize=True)
            #     print("noise_vector_batch:", noise_vector_batch)

            # fake_img_224 = F.interpolate(fake_img_batch, size=(224, 224), mode='bilinear', align_corners=True)
            # fake_img_224.require_grad = True

            fake_img_batch.require_grad = True

            # _fake_img_224 = fake_img_224.view(fake_img_224.size(0), -1)
            # f_mean = _fake_img_224.mean(axis=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(fake_img_224.size(0), 3, 224, 224)
            # f_std = _fake_img_224.std(axis=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(fake_img_224.size(0), 3, 224, 224)
            # fake_img_norm = (fake_img_224 - f_mean) / f_std
            # fake_img_norm = fake_img_norm.cuda()
            # fake_img_norm.require_grad = True

            _fake_img = fake_img_batch.view(fake_img_batch.size(0), -1)
            f_mean = _fake_img.mean(axis=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(fake_img_batch.size(0), 3, 32, 32)
            f_std = _fake_img.std(axis=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(fake_img_batch.size(0), 3, 32, 32)
            fake_img_norm = (fake_img_batch - f_mean) / f_std
            fake_img_norm = fake_img_norm.cuda()
            fake_img_norm.require_grad = True

            feature_fake_img_batch = model(fake_img_norm)
            loss_b = mse_loss(feature_fake_img_batch, feature_objective_batch)
            opt2.zero_grad()
            loss_b.backward(retain_graph=True)
            opt2.step()

            loss_file = os.path.join(self.opt.log_path, 'loss.txt')
            if epoch % 10 == 0:
                # print('Step2: Epoch: %d  loss_b: %.5f' % (epoch, loss_b.data.item()))
                fd = open(loss_file, 'a+')
                string = ('Step2: Epoch: {0}\t'
                         'loss_b {1}\t'.format(epoch, loss_b.data.item()))
                # print(string)
                fd.write(string + '\n')
                fd.close()
            '''
        return fake_img_batch


class UnrolledBlackBoxOptimizer2(nn.Module):
    """
    Args:
        - nscale : number of scales
        - alpha : scale factor in the softmax in the expansion (rho in the paper)
        - nblock : number of stages (K in the paper)
        - K : kernel size
    """
    def __init__(self, opt, teacher, student, generator, X, y, proj_matrix=None):
        super(UnrolledBlackBoxOptimizer, self).__init__()

        self.opt = opt
        self.optim_blocks = nn.ModuleList()

        self.loss_fn = nn.MSELoss()
        self.adversarial_loss = nn.BCELoss()

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

            gt_x = self.X[i_min:i_max].cuda()
            gt_y = self.Y[i_min:i_max].cuda()

            generated_y = torch.randint(0, 2, (self.opt.batch_size,), dtype=torch.float).cuda()

            # Sample noise and labels as generator input
            # z = Variable(torch.randn(gt_x.shape)).cuda()
            z = Variable(torch.randn((self.opt.batch_size, self.opt.latent_dim))).cuda()

            # x = torch.cat((w_t, w_t-w_star, z), dim=1)
            x = torch.cat((w_t, z), dim=1)
            generated_x = self.generator(x, generated_y)

            generated_x_proj = generated_x @ self.proj_matrix.cuda()

            # mixup data
            # mixed_x, targets_a, targets_b, lam = mixup_data(gt_x, generated_x, gt_y, generated_y)
            # mixed_x, targets_a, targets_b = map(Variable, (mixed_x, targets_a, targets_b))

            # self.student.train()
            # out = self.student(mixed_x)
            out = self.student(generated_x_proj)

            # loss = mixup_criterion(self.loss_fn, out, targets_a.float(), targets_b.float(), lam)
            loss = self.loss_fn(out, generated_y.float())

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
            # out_stu = self.teacher(generated_x)
            # out_stu = self.student(generated_x)
            # out_stu = self.student(mixed_x)
            # out_stu = self.student(gt_x)

            # loss_stu = loss_stu + tau * self.loss_fn(out_stu, gt_y)
            # loss_stu = loss_stu + tau * mixup_criterion(self.loss_fn, out_stu, targets_a.float(), targets_b.float(), lam)

            out_stu = new_weight @ torch.transpose(gt_x, 0, 1)
            loss_stu = loss_stu + tau * self.loss_fn(out_stu, gt_y)

        # out_stu = new_weight @ torch.transpose(gt_x, 0, 1)
        # loss_stu = self.loss_fn(out_stu, gt_y)

        w_t = self.student.lin.weight

        i = torch.randint(0, self.nb_batch, size=(1,)).item()
        i_min = i * self.opt.batch_size
        i_max = (i + 1) * self.opt.batch_size

        gt_x = self.X[i_min:i_max].cuda()
        gt_y = self.Y[i_min:i_max].cuda()

        # z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, gt_x.shape)))
        # z = Variable(torch.randn(gt_x.shape)).cuda()
        z = Variable(torch.randn((self.opt.batch_size, self.opt.latent_dim))).cuda()

        # x = torch.cat((w_t, w_t-w_star, gt_x, generated_labels.unsqueeze(0)), dim=1)
        # x = torch.cat((w_t, w_t-w_star, z), dim=1)
        x = torch.cat((w_t, z), dim=1)
        generated_x = self.generator(x, gt_y)

        # generated_labels = generated_labels.float()
        validity = netD(generated_x, Variable(gt_y.type(torch.cuda.LongTensor)))
        g_loss = self.adversarial_loss(validity, valid)

        alpha = 1
        # loss_stu = loss_stu / (self.opt.n_unroll_blocks * alpha)
        loss_stu = loss_stu * alpha
        loss_final = loss_stu + g_loss

        grad_stu = torch.autograd.grad(outputs=loss_final,
                                       inputs=model_paramters,
                                       create_graph=True, retain_graph=True)

        return grad_stu, loss_stu, generated_x, gt_y, g_loss
