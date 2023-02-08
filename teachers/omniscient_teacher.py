# Some code is taken from the following Github repository:
# https://github.com/Ipsedo/IterativeMachineTeaching

from teachers.utils import BaseLinear, BaseConv
import torch
import sys
import torch.nn as nn

import numpy as np


def __example_difficulty__(student, X, y):
    """
    Retourne la difficulté de l'exemple (X, y) selon le student
    :param student: Student ayant un attribut "lin" de class torch.nn.Linear
    :param X: La donnée
    :param y: Le label de la donnée
    :return: Le score de difficulté de l'exemple (X, y)
    """
    # We want to be able to calculate the gradient -> train()
    student.train()

    # Zeroing the accumulated gradient on the student's weights
    student.optim.zero_grad()

    # We want to retain the weight gradient of the linear layer lin
    # student.lin.weight.retain_grad()

    out = student(X)
    loss = student.loss_fn(out.squeeze(1), y)
    loss.backward()

    # layer gradient recovery
    res = student.lin.weight.grad

    # returns the norm of the squared gradient
    return (torch.linalg.norm(res, ord=2) ** 2).item()


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
    loss = student.loss_fn(out.squeeze(1), y)

    loss.backward()

    # layer gradient recovery
    res = student.lin.weight.grad

    # produit scalaire entre la différence des poids et le gradient du student
    return torch.dot(diff.view(-1), res.view(-1)).item()


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

    min_score = sys.float_info.max
    arg_min = 0

    # TODO
    # - one "forward" scoring pass
    # - sort n * log(n)
    # - get first examples

    for i in range(nb_batch):
        i_min = i * batch_size
        i_max = (i + 1) * batch_size

        data = X[i_min:i_max]
        label = y[i_min:i_max]

        lr = student.optim.param_groups[0]["lr"]

        # Calculate the score per batch
        s = (lr ** 2) * student.example_difficulty(data, label)
        s -= lr * 2 * student.example_usefulness(teacher.lin.weight, data, label)

        if s < min_score:
            min_score = s
            arg_min = i

    return arg_min


def __select_example_random_label__(teacher, student, X, y, batch_size):
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

    min_score = sys.float_info.max
    arg_min = 0

    # TODO
    # - one "forward" scoring pass
    # - sort n * log(n)
    # - get first examples

    random_label = torch.from_numpy(np.random.randint(0, 2, batch_size)).cuda()

    for i in range(nb_batch):
        i_min = i * batch_size
        i_max = (i + 1) * batch_size

        data = X[i_min:i_max]
        label = y[i_min:i_max]

        if random_label == label:
            lr = student.optim.param_groups[0]["lr"]

            # Calculate the score per batch
            s = (lr ** 2) * student.example_difficulty(data, label)
            s -= lr * 2 * student.example_usefulness(teacher.lin.weight, data, label)

            if s < min_score:
                min_score = s
                arg_min = i

    return arg_min


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
    def select_example(self, student, X, y, batch_size):
        return __select_example__(self, student, X, y, batch_size)

    def select_example_random_label(self, student, X, y, batch_size):
        return __select_example_random_label__(self, student, X, y, batch_size)


class OmniscientConvTeacher(BaseConv):
    """
    Omnsicient teacher
    Pour un classifieur à convolution de classe OmniscientConvStudent
    """
    def select_example(self, student, X, y, batch_size):
        return __select_example__(self, student, X, y, batch_size)
