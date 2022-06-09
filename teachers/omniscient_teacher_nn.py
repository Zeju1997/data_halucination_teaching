from teachers.utils import BaseLinear, BaseConv
import torch
import sys
import torch.nn as nn
from torch.autograd import grad
from torch.autograd import Variable
import scipy.optimize as spo
from torch.autograd.functional import hessian
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt

import numpy_ml.neural_nets.schedulers as schedulers


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
    loss.backward(retain_graph=True)

    # test = grad(loss, X)

    # layer gradient recovery
    # res = student.lin.weight.grad
    # res_difficulty = Variable(student.lin.weight.grad, requires_grad=True)
    # res_difficulty = torch.clone(student.lin.weight.grad)
    res_difficulty = student.lin.weight.grad
    res_difficulty.requires_grad = True

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

    loss.backward(retain_graph=True)

    # layer gradient recovery
    # res = student.lin.weight.grad
    res_useful = Variable(student.lin.weight.grad, requires_grad=True)

    example_usefulness_loss = torch.dot(diff.view(-1), res_useful.view(-1))

    # produit scalaire entre la différence des poids et le gradient du student
    # return torch.dot(diff.view(-1), res.view(-1)).item()

    return example_usefulness_loss


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


def __generate_example__(teacher, student, X, y, batch_size, lr_factor, gd_n):
    """
    Selectionne un exemple selon le teacher et le student
    :param teacher: Le teacher de classe mère BaseLinear
    :param student: Le student devant implémenter les deux méthodes example_difficulty et example_usefulness
    :param X: Les données
    :param y: les labels des données
    :param batch_size: La taille d'un batch de données
    :return: L'indice de l'exemple à enseigner au student
    """

    #inputs = (torch.rand(2), torch.rand(2))
    #score_loss = ScoreLoss()
    #test = torch.autograd.functional.hessian(score_loss, inputs)

    nb_example = X.size(0)
    nb_batch = int(nb_example / batch_size)

    # TODO
    # - one "forward" scoring pass
    # - sort n * log(n)
    # - get first examples
    # data = Variable(torch.rand(batch_size, 2).cuda(), requires_grad=True)
    # teacher.requires_grad_(False)
    # label = y

    lr = student.optim.param_groups[0]["lr"]

    # big = torch.max(X, 0)
    # small = torch.min(X, 0)

    # data = (X.max() - X.min()) * torch.rand(batch_size, X.size(1)).cuda() + X.min()
    data = torch.rand(batch_size, X.size(1)).cuda()

    # eps = np.sqrt(np.finfo(float).eps)
    # eps = np.array(eps)
    # eps = torch.from_numpy(eps)

    # test = optimize.approx_fprime(x_start, score_loss, [eps, np.sqrt(200) * eps])
    # test = approx_fprime(x_start, score_loss, [eps, np.sqrt(200) * eps])
    s1 = []
    min_score = sys.float_info.max
    arg_min = 0
    num = 0

    for _ in range(gd_n):
        label = torch.randint(0, 1, (batch_size,), dtype=torch.float).cuda()

        # test = student.example_difficulty(data, label)

        example_difficulty = ExampleDifficulty(student, lr, label[0])
        example_usefulness = ExampleUsefulness(student, teacher, lr, label[0])

        score_loss = ScoreLoss(example_difficulty, example_usefulness)

        eps = np.sqrt(np.finfo(float).eps)
        # grad = approx_fprime(data, score_loss, [eps, np.sqrt(200) * eps])
        grad = approx_fprime(data, score_loss, [np.sqrt(200) * eps, np.sqrt(200) * eps])
        # grad = approx_fprime(data, score_loss, [np.sqrt(200) * eps] * data.size(1))
        grad = torch.Tensor(grad).cuda()

        norm_factor = torch.norm(grad)

        data = data - lr * grad * lr_factor

        s = score_loss(data)
        s1.append(s)

        if norm_factor == 0:
            break

    visualize = False
    if visualize:
        fig = plt.figure(figsize=(8,5))
        plt.plot(s1, color="b")
        plt.title('Gaussian Data')
        plt.show()

    min_score = 1000 # sys.float_info.max
    arg_min = 0
    label = y
    best_data = 0
    best_label = 0
    for i in range(nb_batch):
        i_min = i * batch_size
        i_max = (i + 1) * batch_size

        data = X[i_min:i_max]
        label = y[i_min:i_max]

        # Calculate the score per batch
        lr = student.optim.param_groups[0]["lr"]

        example_difficulty = ExampleDifficulty(student, lr, label)
        example_usefulness = ExampleUsefulness(student, teacher, lr, label)
        score_loss = ScoreLoss(example_difficulty, example_usefulness)

        s2 = score_loss(data)
        
        # s2 = (lr ** 2) * student.example_difficulty(data, label)
        # s2 -= lr * 2 * student.example_usefulness(teacher.lin.weight, data, label)
        
        if s2 < min_score:
            min_score = s2
            arg_min = i
            best_data = data
            best_label = label
            # print(s1-s)

            # print("arg min", arg_min, "s", s)

    small = (s < min_score.item())
    print("min score", min_score, "s", s, "smaller", small)

    return best_data, best_label, data


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
    def generate_example(self, student, X, y, batch_size, lr_factor, gd_n):
        return __generate_example__(self, student, X, y, batch_size, lr_factor, gd_n)


class OmniscientConvTeacher(BaseConv):
    """
    Omnsicient teacher
    Pour un classifieur à convolution de classe OmniscientConvStudent
    """
    def select_example(self, student, X, y, batch_size):
        return __select_example__(self, student, X, y, batch_size)
