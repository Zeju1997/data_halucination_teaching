from teachers.utils import BaseLinear, BaseConv
import torch
import sys
import torch.nn as nn
from torch.autograd import grad
from torch.autograd import Variable
import scipy.optimize as spo
from torch.autograd.functional import hessian
from scipy import optimize

import numdifftools as nd

import numpy as np
import matplotlib.pyplot as plt

# from mpl_toolkits.mplot3d import Axes3D

# import numpy_ml.neural_nets.schedulers as schedulers

from tqdm import tqdm


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def __example_difficulty__(student, X, y):
    """
    Retourne la difficulté de l'exemple (X, y) selon le student
    :param student: Student ayant un attribut "lin" de class torch.nn.Linear
    :param X: La donnée
    :param y: Le label de la donnée
    :return: Le score de difficulté de l'exemple (X, y)
    """
    '''
    inp = Variable(torch.rand(3, 4), requires_grad=True)
    W = Variable(torch.rand(4, 4), requires_grad=True)
    yreal = Variable(torch.rand(3, 4), requires_grad=False)
    gradsreal = Variable(torch.rand(3, 4), requires_grad=True)

    print("1", inp.grad)
    ypred = torch.matmul(inp, W)
    ypred.backward(torch.ones(ypred.shape), retain_graph=True)
    print("2", inp.grad)
    gradspred, = grad(ypred, inp,
                      grad_outputs=ypred.data.new(ypred.shape).fill_(1),
                      create_graph=True,
                      retain_graph=True)
    print("3", inp.grad)
    loss = torch.mean((yreal - ypred) ** 2 + (gradspred - gradsreal) ** 2)
    loss.backward()
    print("4", inp.grad)
    '''

    '''
    inp = Variable(torch.rand(1, 2), requires_grad=True)
    W = Variable(torch.rand(2, 1), requires_grad=True)
    yreal = Variable(torch.rand(3, 4), requires_grad=False)
    gradsreal = Variable(torch.rand(3, 4), requires_grad=True)

    print("1", inp.grad)
    ypred = torch.matmul(inp, W)

    gradspred_W, = grad(ypred, W,
                  grad_outputs=ypred.data.new(ypred.shape).fill_(1),
                  create_graph=True,
                  retain_graph=True)

    gradspred_i, = grad(gradspred_W, inp,
              grad_outputs=ypred.data.new(ypred.shape).fill_(1),
              create_graph=True,
              retain_graph=True)


    ypred.backward(torch.ones(ypred.shape), retain_graph=True)
    print("2", inp.grad)
    gradspred_inp, = grad(ypred, inp,
                      grad_outputs=ypred.data.new(ypred.shape).fill_(1),
                      create_graph=True,
                      retain_graph=True)
    print("3", inp.grad)
    loss = torch.mean((yreal - ypred) ** 2 + (gradspred - gradsreal) ** 2)
    loss.backward()
    print("4", inp.grad)
    '''

    # We want to be able to calculate the gradient -> train()
    student.train()

    # Zeroing the accumulated gradient on the student's weights
    student.optim.zero_grad()

    # We want to retain the weight gradient of the linear layer lin
    # student.lin.weight.retain_grad()
    # X.requires_grad = True
    out = student(X)

    loss = student.loss_fn(out, y)
    loss.backward(retain_graph=True, create_graph=True)

    # test = grad(loss, X)

    # layer gradient recovery
    # res = student.lin.weight.grad
    # res_difficulty = Variable(student.lin.weight.grad, requires_grad=True)
    # res_difficulty = torch.clone(student.lin.weight.grad)
    res_difficulty = student.lin.weight.grad

    # res_difficulty.requires_grad = True

    example_difficulty_loss = torch.linalg.norm(res_difficulty, ord=2) ** 2
    # test = grad(example_difficulty_loss, X)# , create_graph=True)
    # example_difficulty_loss.backward()# create_graph=True, retain_graph=True)

    # returns the norm of the squared gradient
    # return (torch.linalg.norm(res, ord=2) ** 2).item()

    return example_difficulty_loss


def __example_usefulness__(student, w_star, X, y):
    """
    Retourne l'utilité de l'exemple (X, y) selon le student et les poids du teacher
    :param student: Student ayant un attribut "lin" de class torch.nn.Linear
    :param w_star: Les poids du teacher (hypothèse  objectif)
    :param X: La donnée
    :param y: Le label de la donnée
    :return: Le score d'utilité de l'exemple (X, y)
    """
    # différence des poids entre le student et le teacher
    diff = student.lin.weight - w_star

    # We want to be able to calculate the gradient -> train()
    student.train()

    # Zeroing the accumulated gradient on the student's weights
    student.optim.zero_grad()

    # We want to retain the weight gradient of the linear layer lin
    # student.lin.weight.retain_grad()

    out = student(X)

    loss = student.loss_fn(out, y)

    loss.backward(retain_graph=True, create_graph=True)

    # layer gradient recovery
    res = student.lin.weight.grad
    # res_useful = Variable(student.lin.weight.grad, requires_grad=True)

    example_usefulness_loss = torch.dot(diff.view(-1), res.view(-1))

    # produit scalaire entre la différence des poids et le gradient du student
    # return torch.dot(diff.view(-1), res.view(-1)).item()

    return example_usefulness_loss


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
    def __init__(self, student, lr, label):
        super(ExampleDifficulty, self).__init__()
        self.lr = lr
        self.student = student
        self.label = label

    def forward(self, input):
        return (self.lr ** 2) * self.student.example_difficulty(input, self.label)


class ExampleUsefulness(nn.Module):
    def __init__(self, student, teacher, lr, label):
        super(ExampleUsefulness, self).__init__()
        self.lr = lr
        self.student = student
        self.label = label
        self.teacher = teacher

    def forward(self, input):
        return self.lr * 2 * self.student.example_usefulness(self.teacher.lin.weight, input, self.label)


class ScoreLoss(nn.Module):
    def __init__(self, example_difficulty, example_usefulness):
        super(ScoreLoss, self).__init__()
        self.example_usefulness = example_usefulness
        self.example_difficulty = example_difficulty

    def forward(self, data):
        # data = torch.Tensor(data).cuda()
        score_loss = self.example_difficulty(data) - self.example_usefulness(data)
        return score_loss.cpu().detach().numpy()


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


def fun(x, y):
    return x**2 + y


# objective function
def objective(x, y):
    return x**2.0 + y**2.0


# derivative of objective function
def derivative(x, y):
    return np.asarray([x * 2.0, y * 2.0])


def rosen(x):
    return (1-x[0])**2 + 105.*(x[1]-x[0]**2)**2


def __select_example__(teacher, student, X, y, batch_size):
    """
    Selectionne un exemple selon le teacher et le student
    :param teacher: Le teacher de classe mère BaseLinear
    :param student: Le student devant implémenter les deux méthodes example_difficulty et example_usefulness
    :param X: Les données
    :param y: les labels des données
    :param batch_size: La taille d'un batch de données
    :return: L'indice de l'exemple à enseigner au student
    """
    nb_example = X.size(0)
    nb_batch = int(nb_example / batch_size)

    min_score = 1000
    arg_min = 0

    # TODO
    # - one "forward" scoring pass
    # - sort n * log(n)
    # - get first examples

    best_data = 0
    best_label = 0

    for i in range(nb_batch):
        # i = torch.randint(0, X.size(0), size=(1,)).item()
        # i = t

        i_min = i * batch_size
        i_max = (i + 1) * batch_size

        data = X[i_min:i_max]
        label = y[i_min:i_max].unsqueeze(0)
        # label = label_new

        # Calculate the score per batch
        lr = student.optim.param_groups[0]["lr"]

        example_difficulty = ExampleDifficulty(student, lr, label)
        example_usefulness = ExampleUsefulness(student, teacher, lr, label)
        score_loss = ScoreLoss(example_difficulty, example_usefulness)

        s2 = score_loss(data) # .cpu().detach().numpy()

        # if i == t:
        #     s3 = s2

        # s2 = (lr ** 2) * student.example_difficulty(data, label)
        # s2 -= lr * 2 * student.example_usefulness(teacher.lin.weight, data, label)

        # zzz.append(s2)
        # xxx.append(data[0, 0].cpu())
        # yyy.append(data[0, 1].cpu())

        if s2 < min_score:
            min_score = s2
            arg_min = i
            best_data = data
            best_label = label # .squeeze(0)
            # print("arg min", arg_min, "s", min_score)

    # print("arg min", arg_min, "s", s)

    # return best_data, best_label
    return arg_min


def __generate_example__working__(teacher, student, X, y, batch_size, lr_factor, gd_n, t, optim):
    """
    Selectionne un exemple selon le teacher et le student
    :param teacher: Le teacher de classe mère BaseLinear
    :param student: Le student devant implémenter les deux méthodes example_difficulty et example_usefulness
    :param X: Les données
    :param y: les labels des données
    :param batch_size: La taille d'un batch de données
    :return: L'indice de l'exemple à enseigner au student
    """

    nb_example = X.size(0)
    nb_batch = int(nb_example / batch_size)

    s = 1000

    best_score = 1000
    count = 0

    alpha = 0.02
    beta1 = 0.8
    beta2 = 0.999
    # eps = 1e-8

    bounds = [[X.min().cpu(), X.max().cpu()]] * X.shape[1]
    bounds = np.asarray(bounds)
    # bounds = np.asarray([[X.min().cpu(), X.max().cpu()], [X.min().cpu(), X.max().cpu()]])
    constraints = [False] * bounds.shape[0]

    # generate an initial point
    # data_new1 = torch.rand(batch_size, X.size(1)).cuda() * 4 - 2
    label_new = torch.randint(0, 2, (batch_size,), dtype=torch.float).cuda()
    # run the gradient descent updates

    s1 = []
    s_min = 1000
    count = 0
    zz = []
    xx = []
    yy = []
    data_trajectory = []
    init_point = torch.zeros(batch_size, X.shape[1]).cuda()
    # init_point.requires_grad = True
    # optim = torch.optim.Adam([init_point], lr=0.005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    # init_point = torch.ones(batch_size, X.shape[1]).cuda() * X.min()
    # init_point = X.mean(dim=0).unsqueeze(0)

    diff = X.max(dim=0).values - X.min(dim=0).values

    # init_point = (X.max() - X.min()) * torch.rand(batch_size, X.size(1)).cuda() + X.min()
    # init_point = (X.max(dim=0).values - X.min(dim=0).values) * torch.rand(batch_size, X.size(1)).cuda() + X.min(dim=0).values

    # initialize first and second moments
    m = [torch.zeros(1).cuda() for _ in range(bounds.shape[0])]
    v = [torch.zeros(1).cuda() for _ in range(bounds.shape[0])]
    vhat = [torch.zeros(1).cuda() for _ in range(bounds.shape[0])]

    s1 = []
    for t in range(gd_n):
        lr = student.optim.param_groups[0]["lr"]
        example_difficulty = ExampleDifficulty(student, lr, label_new)
        example_usefulness = ExampleUsefulness(student, teacher, lr, label_new)

        # TODO: Try use autograd
        '''
        optim.zero_grad()
        score_loss = example_difficulty(init_point) + example_usefulness(init_point)
        score_loss.backward()
        s1.append(score_loss.item())
        optim.step()

        point = init_point.detach().clone().cpu().numpy()
        data_trajectory.append(point)
        xx.append(point[0, 0])
        yy.append(point[0, 1])
        zz.append(score_loss.item())
        '''

        score_loss = ScoreLoss(example_difficulty, example_usefulness)

        eps = np.sqrt(np.finfo(float).eps)
        eps_list = [np.sqrt(200) * eps] * X.shape[1]
        grad = approx_fprime(init_point, score_loss, eps_list)
        grad = torch.Tensor(grad).cuda()

        # build a solution one variable at a time
        for i in range(bounds.shape[0]):
            if not constraints[i]:
                if optim == "adam":
                    # adam: convergence problem!
                    # m(t) = beta1 * m(t-1) + (1 - beta1) * g(t)
                    m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i]
                    # v(t) = beta2 * v(t-1) + (1 - beta2) * g(t)^2
                    v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i]**2
                    # mhat(t) = m(t) / (1 - beta1(t))
                    mhat = m[i] / (1.0 - beta1**(t+1))
                    # vhat(t) = v(t) / (1 - beta2(t))
                    vhat = v[i] / (1.0 - beta2**(t+1))
                    # x(t) = x(t-1) - alpha * mhat(t) / (sqrt(vhat(t)) + ep)
                    update = torch.Tensor([alpha]).cuda() * mhat / (torch.sqrt(vhat) + eps)

                else:
                    # AMSGrad
                    # m(t) = beta1(t) * m(t-1) + (1 - beta1(t)) * g(t)
                    m[i] = beta1**(t+1) * m[i] + (1.0 - beta1**(t+1)) * grad[i]
                    # v(t) = beta2 * v(t-1) + (1 - beta2) * g(t)^2
                    v[i] = (beta2 * v[i]) + (1.0 - beta2) * grad[i]**2
                    # vhat(t) = max(vhat(t-1), v(t))
                    vhat[i] = max(vhat[i], v[i])
                    # x(t) = x(t-1) - alpha(t) * m(t) / sqrt(vhat(t)))
                    update = torch.Tensor([alpha]).cuda() * m[i] / (torch.sqrt(vhat[i]) + 1e-8)

                # escape local minima?
                #if torch.norm(grad) == 0:
                #    noise = torch.empty(1).normal_(mean=0, std=0.1).cuda()
                #    update = update + noise

                if constraints[i]:
                    update[0] = 0

                init_point[0, i] = init_point[0, i] - update
                # print("update", update)

                if init_point[0, i] > X.max() or init_point[0, i] < X.min():
                    constraints[i] = True

        s = score_loss(init_point)

        if len(s1) != 0:
            if s == s1[-1]:
                count = count + 1
            else:
                count = 0

        if count > 10:
            break

        s1.append(s)

        zz.append(s)
        xx.append(init_point[0, 0].cpu())
        yy.append(init_point[0, 1].cpu())

        data_trajectory.append(init_point)

    s1_np = np.array(s1)
    idx = np.argmin(s1_np)
    data_new = data_trajectory[idx]

    visualize = False
    if visualize:
        z = []
        x = np.linspace(X.min().cpu(), X.max().cpu(), 200)
        y = np.linspace(X.min().cpu(), X.max().cpu(), 200)
        for i in tqdm(range(200)):
            for j in range(200):
                grid_point = torch.tensor([x[j], y[i]], dtype=torch.float).cuda()
                example_difficulty = ExampleDifficulty(student, lr, label_new)
                example_usefulness = ExampleUsefulness(student, teacher, lr, label_new)

                score_loss = ScoreLoss(example_difficulty, example_usefulness)

                s = score_loss(grid_point)
                z.append(s)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x_grid, y_grid = np.meshgrid(x, y)
        zs = np.array(z)
        z_grid = zs.reshape(x_grid.shape)

        ax.plot_surface(x_grid, y_grid, z_grid)
        ax.scatter(np.array(xx), np.array(yy), np.array(zz), color='r', alpha=1)
        # ax.scatter(np.array(xxx), np.array(yyy), np.array(zzz), color='k', alpha=1)
        # ax.scatter(best_data[0, 0].cpu().numpy(), best_data[0, 1].cpu().numpy(), min_score, color='g', alpha=1)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()

    visualize = False
    if visualize:
        fig = plt.figure(figsize=(8, 5))
        plt.plot(s1, color="b")
        plt.title('Optimizer Score')
        plt.show()

    return data_new, label_new


def __generate_example__(teacher, student, X, y, batch_size, lr_factor, gd_n, t, optim):
    """
    Selectionne un exemple selon le teacher et le student
    :param teacher: Le teacher de classe mère BaseLinear
    :param student: Le student devant implémenter les deux méthodes example_difficulty et example_usefulness
    :param X: Les données
    :param y: les labels des données
    :param batch_size: La taille d'un batch de données
    :return: L'indice de l'exemple à enseigner au student
    """

    nb_example = X.size(0)
    nb_batch = int(nb_example / batch_size)

    s = 1000

    best_score = 1000
    count = 0

    alpha = 0.02
    beta1 = 0.8
    beta2 = 0.999
    # eps = 1e-8

    bounds = [[X.min().cpu(), X.max().cpu()]] * X.shape[1]
    bounds = np.asarray(bounds)
    # bounds = np.asarray([[X.min().cpu(), X.max().cpu()], [X.min().cpu(), X.max().cpu()]])
    constraints = [False] * bounds.shape[0]

    # generate an initial point
    # data_new1 = torch.rand(batch_size, X.size(1)).cuda() * 4 - 2
    label_new = torch.randint(0, 2, (batch_size,), dtype=torch.float).cuda()
    # run the gradient descent updates

    s1 = []
    s_min = 1000
    count = 0
    zz = []
    xx = []
    yy = []
    data_trajectory = []
    init_point = torch.zeros(batch_size, X.shape[1]).cuda()
    init_point.requires_grad = True

    # init_point = torch.ones(batch_size, X.shape[1]).cuda() * X.min()
    # init_point = X.mean(dim=0).unsqueeze(0)

    diff = X.max(dim=0).values - X.min(dim=0).values

    # init_point = (X.max() - X.min()) * torch.rand(batch_size, X.size(1)).cuda() + X.min()
    # init_point = (X.max(dim=0).values - X.min(dim=0).values) * torch.rand(batch_size, X.size(1)).cuda() + X.min(dim=0).values

    # initialize first and second moments
    m = [torch.zeros(1).cuda() for _ in range(bounds.shape[0])]
    v = [torch.zeros(1).cuda() for _ in range(bounds.shape[0])]
    vhat = [torch.zeros(1).cuda() for _ in range(bounds.shape[0])]


    s1 = []
    for t in range(gd_n):
        init_point.requires_grad = True
        lr = student.optim.param_groups[0]["lr"]
        example_difficulty = ExampleDifficulty(student, lr, label_new)
        example_usefulness = ExampleUsefulness(student, teacher, lr, label_new)

        score_loss = example_difficulty(init_point) + example_usefulness(init_point)
        # print("score loss", score_loss)

        grad = torch.autograd.grad(outputs=score_loss,
                                   inputs=init_point,
                                   create_graph=False, retain_graph=False)

        grad = - grad[0].detach().squeeze(0)

        init_point.requires_grad = False

        score_loss = ScoreLoss(example_difficulty, example_usefulness)

        # eps = np.sqrt(np.finfo(float).eps)
        # eps_list = [np.sqrt(200) * eps] * X.shape[1]
        # grad = approx_fprime(init_point, score_loss, eps_list)
        # grad = torch.Tensor(grad).cuda()

        # build a solution one variable at a time
        for i in range(bounds.shape[0]):
            if not constraints[i]:
                if optim == "adam":
                    # adam: convergence problem!
                    # m(t) = beta1 * m(t-1) + (1 - beta1) * g(t)
                    m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i]
                    # v(t) = beta2 * v(t-1) + (1 - beta2) * g(t)^2
                    v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i]**2
                    # mhat(t) = m(t) / (1 - beta1(t))
                    mhat = m[i] / (1.0 - beta1**(t+1))
                    # vhat(t) = v(t) / (1 - beta2(t))
                    vhat = v[i] / (1.0 - beta2**(t+1))
                    # x(t) = x(t-1) - alpha * mhat(t) / (sqrt(vhat(t)) + ep)
                    update = torch.Tensor([alpha]).cuda() * mhat / (torch.sqrt(vhat) + eps)

                else:
                    # AMSGrad
                    # m(t) = beta1(t) * m(t-1) + (1 - beta1(t)) * g(t)
                    m[i] = beta1**(t+1) * m[i] + (1.0 - beta1**(t+1)) * grad[i]
                    # v(t) = beta2 * v(t-1) + (1 - beta2) * g(t)^2
                    v[i] = (beta2 * v[i]) + (1.0 - beta2) * grad[i]**2
                    # vhat(t) = max(vhat(t-1), v(t))
                    vhat[i] = max(vhat[i], v[i])
                    # x(t) = x(t-1) - alpha(t) * m(t) / sqrt(vhat(t)))
                    update = torch.Tensor([alpha]).cuda() * m[i] / (torch.sqrt(vhat[i]) + 1e-8)

                # escape local minima?
                #if torch.norm(grad) == 0:
                #    noise = torch.empty(1).normal_(mean=0, std=0.1).cuda()
                #    update = update + noise

                if constraints[i]:
                    update[0] = 0

                init_point[0, i] = init_point[0, i] - update
                # print("update", update)

                if init_point[0, i] > X.max() or init_point[0, i] < X.min():
                    constraints[i] = True

        s = score_loss(init_point)

        if len(s1) != 0:
            if s == s1[-1]:
                count = count + 1
            else:
                count = 0

        if count > 10:
            break

        s1.append(s)

        zz.append(s)
        xx.append(init_point[0, 0].cpu())
        yy.append(init_point[0, 1].cpu())

        data_trajectory.append(init_point)

    s1_np = np.array(s1)
    idx = np.argmin(s1_np)
    data_new = data_trajectory[idx]

    visualize = False
    if visualize:
        z = []
        x = np.linspace(X.min().cpu(), X.max().cpu(), 200)
        y = np.linspace(X.min().cpu(), X.max().cpu(), 200)
        for i in tqdm(range(200)):
            for j in range(200):
                grid_point = torch.tensor([x[j], y[i]], dtype=torch.float).cuda()
                example_difficulty = ExampleDifficulty(student, lr, label_new)
                example_usefulness = ExampleUsefulness(student, teacher, lr, label_new)

                score_loss = ScoreLoss(example_difficulty, example_usefulness)

                s = score_loss(grid_point)
                z.append(s)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x_grid, y_grid = np.meshgrid(x, y)
        zs = np.array(z)
        z_grid = zs.reshape(x_grid.shape)

        ax.plot_surface(x_grid, y_grid, z_grid)
        ax.scatter(np.array(xx), np.array(yy), np.array(zz), color='r', alpha=1)
        # ax.scatter(np.array(xxx), np.array(yyy), np.array(zzz), color='k', alpha=1)
        # ax.scatter(best_data[0, 0].cpu().numpy(), best_data[0, 1].cpu().numpy(), min_score, color='g', alpha=1)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()

    visualize = False
    if visualize:
        fig = plt.figure(figsize=(8, 5))
        plt.plot(s1, color="b")
        plt.title('Optimizer Score')
        plt.show()


    return data_new, label_new


def __generate_example_both__(teacher, student, X, y, batch_size, lr_factor, gd_n, t):
    """
    Selectionne un exemple selon le teacher et le student
    :param teacher: Le teacher de classe mère BaseLinear
    :param student: Le student devant implémenter les deux méthodes example_difficulty et example_usefulness
    :param X: Les données
    :param y: les labels des données
    :param batch_size: La taille d'un batch de données
    :return: L'indice de l'exemple à enseigner au student
    """

    nb_example = X.size(0)
    nb_batch = int(nb_example / batch_size)

    # lr = student.optim.param_groups[0]["lr"]

    min_score = 1000 # sys.float_info.max
    arg_min = 0
    best_data = 0
    best_label = 0

    s3 = 0
    xxx = []
    yyy = []
    zzz = []

    # student_optim =

    # nb_batch = 1
    for i in range(nb_batch):
        # i = torch.randint(0, X.size(0), size=(1,)).item()
        # i = t

        i_min = i * batch_size
        i_max = (i + 1) * batch_size

        data = X[i_min:i_max]
        label = y[i_min:i_max].unsqueeze(0)
        # label = label_new

        # Calculate the score per batch
        lr = student.optim.param_groups[0]["lr"]

        example_difficulty = ExampleDifficulty(student, lr, label)
        example_usefulness = ExampleUsefulness(student, teacher, lr, label)
        score_loss = ScoreLoss(example_difficulty, example_usefulness)

        s2 = score_loss(data) # .cpu().detach().numpy()

        # if i == t:
        #     s3 = s2

        # s2 = (lr ** 2) * student.example_difficulty(data, label)
        # s2 -= lr * 2 * student.example_usefulness(teacher.lin.weight, data, label)

        # zzz.append(s2)
        # xxx.append(data[0, 0].cpu())
        # yyy.append(data[0, 1].cpu())

        if s2 < min_score:
            min_score = s2
            arg_min = i
            best_data = data
            best_label = label
            # print("arg min", arg_min, "s", min_score)

    s = 1000
    s1 = []
    better = (s < min_score.item())

    zz = []
    xx = []
    yy = []
    best_score = 1000
    count = 0

    alpha = 0.02
    beta1 = 0.8
    beta2 = 0.999
    # eps = 1e-8

    bounds = [[X.min().cpu(), X.max().cpu()]] * X.shape[1]
    bounds = np.asarray(bounds)
    # bounds = np.asarray([[X.min().cpu(), X.max().cpu()], [X.min().cpu(), X.max().cpu()]])
    constraints = [False] * bounds.shape[0]

    # generate an initial point
    # data_new = (X.max() - X.min()) * torch.rand(batch_size, X.size(1)).cuda() + X.min()
    # data_new1 = torch.rand(batch_size, X.size(1)).cuda() * 4 - 2
    data_new = torch.zeros(1, X.shape[1]).cuda()
    label_new = torch.randint(0, 2, (batch_size,), dtype=torch.float).cuda()

    # initialize first and second moments
    m = [0.0 for _ in range(bounds.shape[0])]
    v = [0.0 for _ in range(bounds.shape[0])]
    vhat = [0.0 for _ in range(bounds.shape[0])]
    # run the gradient descent updates
    for t in range(gd_n):
        lr = student.optim.param_groups[0]["lr"]
        example_difficulty = ExampleDifficulty(student, lr, label_new)
        example_usefulness = ExampleUsefulness(student, teacher, lr, label_new)

        score_loss = ScoreLoss(example_difficulty, example_usefulness)

        # score = score_loss(data_new)

        eps = np.sqrt(np.finfo(float).eps)
        eps_list = [np.sqrt(200) * eps] * X.shape[1]
        grad = approx_fprime(data_new, score_loss, eps_list)
        grad = torch.Tensor(grad).cuda()

        # grad = nd.Gradient(score_loss)(data_new.cpu().detach().numpy())
        # grad = torch.Tensor(grad).cuda()

        # print(t, grad)

        # build a solution one variable at a time
        for i in range(bounds.shape[0]):
            if not constraints[i]:
                # adam: convergence problem!
                # m(t) = beta1 * m(t-1) + (1 - beta1) * g(t)
                m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i]
                # v(t) = beta2 * v(t-1) + (1 - beta2) * g(t)^2
                v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i]**2
                # mhat(t) = m(t) / (1 - beta1(t))
                mhat = m[i] / (1.0 - beta1**(t+1))
                # vhat(t) = v(t) / (1 - beta2(t))
                vhat = v[i] / (1.0 - beta2**(t+1))
                # x(t) = x(t-1) - alpha * mhat(t) / (sqrt(vhat(t)) + ep)
                update = torch.Tensor([alpha]).cuda() * mhat / (torch.sqrt(vhat) + eps)

                '''
                # AMSGrad
                # m(t) = beta1(t) * m(t-1) + (1 - beta1(t)) * g(t)
                m[i] = beta1**(t+1) * m[i] + (1.0 - beta1**(t+1)) * grad[i]
                # v(t) = beta2 * v(t-1) + (1 - beta2) * g(t)^2
                v[i] = (beta2 * v[i]) + (1.0 - beta2) * grad[i]**2
                # vhat(t) = max(vhat(t-1), v(t))
                vhat[i] = max(vhat[i], v[i])
                # x(t) = x(t-1) - alpha(t) * m(t) / sqrt(vhat(t)))
                update = torch.Tensor([alpha]).cuda() * m[i] / (torch.sqrt(vhat[i]) + 1e-8)
                '''

                # escape local minima?
                if torch.norm(grad) == 0:
                    noise = torch.empty(1).normal_(mean=0, std=0.01).cuda()
                    update = update + noise

                if constraints[i]:
                    update[0] = 0

                data_new[0, i] = data_new[0, i] - update
                # print("update", update)

                if data_new[0, i] > X.max() or data_new[0, i] < X.min():
                    constraints[i] = True

        s = score_loss(data_new)
        s1.append(s)

        zz.append(s)
        xx.append(data_new[0, 0].cpu())
        yy.append(data_new[0, 1].cpu())

    print("c")

    better = min_score > s
    print("min score", min_score, "s", s, "better", better)

    visualize = False
    if visualize:
        z = []
        x = np.linspace(X.min().cpu(), X.max().cpu(), 200)
        y = np.linspace(X.min().cpu(), X.max().cpu(), 200)
        for i in tqdm(range(200)):
            for j in range(200):
                grid_point = torch.tensor([x[j], y[i]], dtype=torch.float).cuda()
                example_difficulty = ExampleDifficulty(student, lr, label_new)
                example_usefulness = ExampleUsefulness(student, teacher, lr, label_new)

                score_loss = ScoreLoss(example_difficulty, example_usefulness)

                s = score_loss(grid_point)
                z.append(s)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x_grid, y_grid = np.meshgrid(x, y)
        zs = np.array(z)
        z_grid = zs.reshape(x_grid.shape)

        ax.plot_surface(x_grid, y_grid, z_grid)
        ax.scatter(np.array(xx), np.array(yy), np.array(zz), color='r', alpha=0.5)
        ax.scatter(np.array(xxx), np.array(yyy), np.array(zzz), color='k', alpha=1)
        ax.scatter(best_data[0, 0].cpu().numpy(), best_data[0, 1].cpu().numpy(), min_score, color='g', alpha=1)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()

    visualize = False
    if visualize:
        fig = plt.figure(figsize=(8, 5))
        plt.plot(s1, color="b")
        plt.title('Optimizer Score')
        plt.show()

    # return best_data, best_label, data_new, label_new, better
    return best_data, best_label, best_data, best_label, better
    # return data_new, label_new, data_new, label_new, better


class OmniscientLinearStudent(BaseLinear):
    """
    Classe pour le student du omniscient teacher
    Classification linéaire
    Marche de paire avec OmniscientLinearTeacher
    """
    def example_difficulty(self, X, y):
        return __example_difficulty__(self, X, y)

    def example_usefulness(self, w_star, X, y):
        return __example_usefulness__(self, w_star, X, y)

    def get_weight_grad(self, X, y):
        return __get_weight_grad__(self, X, y)


class OmniscientConvStudent(BaseConv):
    """
    Classe pour le student du omniscient teacher
    Modèle à convolution.
    Marche de paire avec OmniscientConvTeacher
    """
    def example_difficulty(self, X, y):
        return __example_difficulty__(self, X, y)

    def example_usefulness(self, w_star, X, y):
        return __example_usefulness__(self, w_star, X, y)


class OmniscientLinearTeacher(BaseLinear):
    """
    Omniscient teacher.
    Pour un classifieur linéaire de classe OmniscientLinearStudent
    """
    def generate_example(self, student, X, y, batch_size, lr_factor, gd_n, t, optim):
        return __generate_example__(self, student, X, y, batch_size, lr_factor, gd_n, t, optim)

    def select_example(self, student, X, y, batch_size):
        return __select_example__(self, student, X, y, batch_size)


class OmniscientConvTeacher(BaseConv):
    """
    Omnsicient teacher
    Pour un classifieur à convolution de classe OmniscientConvStudent
    """
    def select_example(self, student, X, y, batch_size):
        return __generate_example__(self, student, X, y, batch_size)
