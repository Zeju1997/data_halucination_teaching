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


def approx_fprime(xk, f, epsilon, args=(), f0=None):
    """
    See ``approx_fprime``.  An optional initial function value arg is added.

    """
    if f0 is None:
        f0 = f(*((xk,) + args))
    grad = np.zeros((xk.shape[1],), float)
    # grad = torch.zeros(len(xk),).cuda()
    ei = np.zeros((xk.shape[1],), float)
    # ei = torch.zeros(len(xk),).cuda()
    for k in range(xk.shape[1]):
        ei[k] = 1.0
        d = epsilon * ei
        d = torch.Tensor(d).cuda()
        grad[k] = (f(*((xk + d,) + args)) - f0) / d[k]
        ei[k] = 0.0
    return grad


def __get_weight_grad__(student, X, y):
    student.train()

    # Zeroing the accumulated gradient on the student's weights
    student.optim.zero_grad()

    # We want to retain the weight gradient of the linear layer lin
    # student.lin.weight.retain_grad()
    # X.requires_grad = True
    out = student(X)
    loss = student.loss_fn(out, y)
    loss.backward()

    res = student.lin.weight.grad
    return res


class ExampleDifficulty(nn.Module):
    def __init__(self, student, loss_fn, lr, label):
        super(ExampleDifficulty, self).__init__()
        self.lr = lr
        self.student = student
        self.label = label
        self.loss_fn = loss_fn

    def forward(self, input):
        return (self.lr ** 2) * self.example_difficulty(self.student, input, self.label)

    def example_difficulty(self, student, X, y):
        # We want to be able to calculate the gradient -> train()
        student.train()

        # Zeroing the accumulated gradient on the student's weights
        # student.optim.zero_grad()
        student.zero_grad()

        # We want to retain the weight gradient of the linear layer lin
        # student.lin.weight.retain_grad()
        # X.requires_grad = True
        out = student(X)

        loss = self.loss_fn(out, y)
        # loss.backward(create_graph=True, retain_graph=True)

        grad = torch.autograd.grad(outputs=loss,
                                    inputs=student.lin.weight,
                                    create_graph=True, retain_graph=True)

        # res_difficulty = student.lin.weight.grad
        res_difficulty = grad[0]

        # res_difficulty.requires_grad = True

        example_difficulty_loss = torch.linalg.norm(res_difficulty, ord=2) ** 2
        # test = grad(example_difficulty_loss, X)# , create_graph=True)
        # example_difficulty_loss.backward()# create_graph=True, retain_graph=True)

        # returns the norm of the squared gradient
        # return (torch.linalg.norm(res, ord=2) ** 2).item()

        return example_difficulty_loss


class ExampleUsefulness(nn.Module):
    def __init__(self, student, teacher, loss_fn, lr, label):
        super(ExampleUsefulness, self).__init__()
        self.lr = lr
        self.student = student
        self.label = label
        self.teacher = teacher
        self.loss_fn = loss_fn

    def forward(self, input):
        return self.lr * 2 * self.example_usefulness(self.student, self.teacher.lin.weight, input, self.label)

    def example_usefulness(self, student, w_star, X, y):
        # différence des poids entre le student et le teacher
        diff = student.lin.weight - w_star

        # We want to be able to calculate the gradient -> train()
        student.train()

        # Zeroing the accumulated gradient on the student's weights
        # student.optim.zero_grad()
        student.zero_grad()

        # We want to retain the weight gradient of the linear layer lin
        # student.lin.weight.retain_grad()

        out = student(X)

        loss = self.loss_fn(out, y)

        # loss.backward(create_graph=False, retain_graph=True)
        grad = torch.autograd.grad(outputs=loss,
                                    inputs=student.lin.weight,
                                    create_graph=True, retain_graph=True)

        # layer gradient recovery
        # res = student.lin.weight.grad
        res = grad[0]

        example_usefulness_loss = torch.dot(diff.view(-1), res.view(-1))

        # produit scalaire entre la différence des poids et le gradient du student
        # return torch.dot(diff.view(-1), res.view(-1)).item()

        return example_usefulness_loss


class ScoreLoss(nn.Module):
    def __init__(self, example_difficulty, example_usefulness):
        super(ScoreLoss, self).__init__()
        self.example_usefulness = example_usefulness
        self.example_difficulty = example_difficulty

    def forward(self, data):
        # data = torch.Tensor(data).cuda()
        score_loss = self.example_difficulty(data) - self.example_usefulness(data)
        # return score_loss.cpu().detach().numpy()
        return score_loss


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
    def __init__(self, opt, loader, fc):
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

        self.fc = fc
        # self.Y = y

        # self.nb_batch = int(self.X.shape[0] / self.opt.batch_size)

        # self.proj_matrix = proj_matrix

    def forward(self, z0, inputs, targets):
        # self.generator.linear.weight = weight
        # self.student.lin.weight = w_init

        with torch.no_grad():
            # for param1 in self.generator.parameters():
            #    param1 = weight
            self.fc.load_state_dict(torch.load(os.path.join(self.opt.log_path, 'tmp_fc.pth')))
            # self.teacher.load_state_dict(torch.load('teacher_wstar.pth'))
            # self.student.load_state_dict(torch.load('teacher_w0.pth'))
            # for param1 in self.student.parameters():
            #     param1 = w_init
            # for param2 in self.teacher.parameters():
            #     param2 = w_star

        loss_stu = 0
        w_loss = 0
        tau = 1

        fc_orig = copy.deepcopy(self.fc)

        pdist = torch.nn.PairwiseDistance(p=2)
        num_steps = 2
        step_size = 0.001
        epsilon = 0.1
        norm = 0.0
        p = 2

        optim = torch.optim.SGD(self.fc.parameters(), lr=0.001)
        num_steps = 5

        # optim_loss = []

        # w = fc.lin.weight
        for n in range(self.opt.n_weight_update):

            optim.zero_grad()
            outputs = self.fc(z0)

            loss = self.loss_fn(outputs, targets)
            loss.backward(retain_graph=True, create_graph=True)

            # optim_loss.append(loss.item())

            optim.step()

        '''
        for n in range(self.opt.n_weight_update):
            try:
                (gt_x, gt_y) = data_iter.next()
            except:
                data_iter = iter(self.loader)
                (gt_x, gt_y) = data_iter.next()

            inputs, targets = gt_x.cuda(), gt_y.long().cuda()

            z = model(inputs)

            optim.zero_grad()
            outputs = fc(z)

            loss = self.loss_fn(outputs, targets)
            loss.backward()

            optim_loss.append(loss.item())

            optim.step()
        '''

        # z = self.update_z(fc, fc_mdl, z0, targets)

        pdist = torch.nn.PairwiseDistance(p=2)

        z = z0
        p = 2
        step_size = 0.02
        epsilon = self.opt.epsilon
        # optim_loss = []

        norm_0 = torch.norm(z0.detach().clone(), p=p)

        example_difficulty = ExampleDifficulty(fc_orig, self.loss_fn, self.opt.lr, targets)
        example_usefulness = ExampleUsefulness(fc_orig, self.fc, self.loss_fn, self.opt.lr, targets)

        for n in range(self.opt.n_z_update):
            loss = example_difficulty(z) - example_usefulness(z)

            gradients = torch.autograd.grad(outputs=loss,
                                            inputs=z,
                                            retain_graph=False, create_graph=False)

            gradients = self.normalize_lp_norms(gradients[0], p=p)
            z = z - step_size * gradients
            z = self.project(z, z0, epsilon, p)
            # diff1 = pdist(z, z0)
            norm_1 = torch.norm(z.detach().clone(), p=p)
            z = z * (norm_0 / norm_1)
            # diff2 = pdist(z, z0)
            # print('diff1', diff1.max(), 'diff2', diff2.max())
            # optim_loss.append(loss.item())

            # print(n, "iter pass", torch.cuda.memory_allocated(0))

        # diff = pdist(z, z0)
        # print('diff', diff.max())

        '''
        z0 = model(inputs)
        z = z0
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
        '''
        '''
        optim.zero_grad()

        outputs = fc(z)
        loss = self.loss_fn(outputs, targets)
        loss.backward()

        optim_loss.append(loss.item())

        optim.step()

        weight = fc.state_dict()
        '''
        # fig = plt.figure()
        # plt.plot(optim_loss, c="b", label="Teacher (CNN)")
        # plt.xlabel("Epoch")
        # plt.ylabel("Accuracy")
        # plt.legend()
        # plt.show()

        del fc_orig

        return z

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

    def forward1(self, model, fc):
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

        # fig = plt.figure()
        # plt.plot(optim_loss, c="b", label="Teacher (CNN)")
        # plt.xlabel("Epoch")
        # plt.ylabel("Accuracy")
        # plt.legend()
        # plt.show()

        return weight

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

    def update_z(self, fc_star, fc, z0, targets):

        # z0 = model(inputs)
        # z0 = z0.detach().clone()
        # z0.requires_grad = True

        pdist = torch.nn.PairwiseDistance(p=2)

        z = z0
        p = 2
        step_size = 0.001
        epsilon = 0.1
        optim_loss = []
        gd_n = 20

        example_difficulty = ExampleDifficulty(fc, self.loss_fn, self.opt.lr, targets)
        example_usefulness = ExampleUsefulness(fc, fc_star, self.loss_fn, self.opt.lr, targets)

        for n in range(gd_n):

            loss = example_difficulty(z) - example_usefulness(z)

            gradients = torch.autograd.grad(outputs=loss,
                                            inputs=z,
                                            retain_graph=False, create_graph=False)

            gradients = self.normalize_lp_norms(gradients[0], p=p)
            z = z - step_size * gradients
            z = self.project(z, z0, epsilon, p)

            optim_loss.append(loss.item())

        # fig = plt.figure()
        # plt.plot(optim_loss, c="b", label="Teacher (CNN)")
        # plt.xlabel("Epoch")
        # plt.ylabel("Accuracy")
        # plt.legend()
        # plt.show()

            # print(n, "iter pass", torch.cuda.memory_allocated(0))

        # print("Before backward pass", torch.cuda.memory_allocated(0))
        # del example_difficulty
        # del example_usefulness
        # print("After backward pass", torch.cuda.memory_allocated(0))

        # z.zero_grad()
        # diff = pdist(z, z0)
        # print(diff.max())

        # delta_z = z - z0

        # return delta_z
        return z


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
