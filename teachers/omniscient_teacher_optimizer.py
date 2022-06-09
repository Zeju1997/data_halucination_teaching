from teachers.utils import BaseLinear, BaseConv
import torch
import sys
import torch.nn as nn
from torch.autograd import grad
from torch.autograd import Variable
import scipy.optimize as spo


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
    X = torch.Tensor(X).cuda()
    # We want to retain the weight gradient of the linear layer lin
    # student.lin.weight.retain_grad()
    X = Variable(torch.cuda.FloatTensor(X))
    out = student(X)
    loss = student.loss_fn(out, y)
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
    X = torch.Tensor(X).cuda()
    # We want to retain the weight gradient of the linear layer lin
    # student.lin.weight.retain_grad()
    X = Variable(torch.cuda.FloatTensor(X))
    out = student(X)
    loss = student.loss_fn(out, y)

    loss.backward()

    # layer gradient recovery
    res = student.lin.weight.grad

    # produit scalaire entre la différence des poids et le gradient du student
    return torch.dot(diff.view(-1), res.view(-1)).item()


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
        return self.example_difficulty(data) - self.example_usefulness(data)


def __generate_example__(teacher, student, X, y, batch_size):
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

    label = torch.randint(0, 1, (batch_size,), dtype=torch.float).cuda()

    example_difficulty = ExampleDifficulty(student, lr, label[0])
    example_usefulness = ExampleUsefulness(student, teacher, lr, label[0])

    score_loss = ScoreLoss(example_difficulty, example_usefulness)
    # loss = example_difficulty(data) - example_usefulness(data) # f(x)
    x_start = torch.rand(batch_size, 2).cpu()

    bnds = [[-2, 2], [-2, 2]]

    const = ({'type': 'ineq', 'fun': lambda x:  x+3},
        {'type': 'ineq', 'fun': lambda x: x-3})

    # result = spo.minimize(score_loss, x_start, method="Nelder-Mead") # Nelder-Mead
    result = spo.minimize(score_loss, x_start, method="COBYLA", bounds=bnds, constraints=const)
    # s1 = score_loss(result.x)
    '''
    rranges = (slice(-2, 2, 0.1), slice(-2, 2, 0.1))
    result = spo.brute(score_loss, rranges, full_output=True,
                          finish=spo.fmin)
    return result[0]
    '''


    '''
    if result.success:
        print("Success!")
        print(f"x = {result.x} y = {result.fun}")
    else:
        print("sorry, could not find a minimum.")
    return result.x
    '''
    '''
    # dx1 = grad(outputs=loss, inputs=data, retain_graph=True)
    loss.backward(retain_graph=True)
    grad1 = data.grad
    print("c", grad1)

    loss = -example_usefulness(data)
    loss.backward(retain_graph=True)
    grad2 = data.grad
    print("grad 2", grad2)
    with torch.no_grad():
        data -= lr * (grad1 + grad2)
        print(data)
    '''

    '''
    min_score = sys.float_info.max
    arg_min = 0
    label = y
    best_data = 0
    for i in range(nb_batch):
        i_min = i * batch_size
        i_max = (i + 1) * batch_size

        data = X[i_min:i_max]
        label = y[i_min:i_max]

        # Calculate the score per batch
        lr = student.optim.param_groups[0]["lr"]

        #example_difficulty = ExampleDifficulty(student, lr, label)
        #example_usefulness = ExampleUsefulness(student, teacher, lr, label)
        #score_loss = ScoreLoss(example_difficulty, example_usefulness)

        #s1 = score_loss(data)
        
        s = (lr ** 2) * student.example_difficulty(data, label)
        s -= lr * 2 * student.example_usefulness(teacher.lin.weight, data, label)
        
        if s < min_score:
            min_score = s
            arg_min = i
            best_data = data
            # print(s1-s)

            # print("arg min", arg_min, "s", s)
    # small = (s1 < min_score)
    # print("min score", min_score, "s1", s1, "smaller", small)
    '''
    return result.x


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
    def generate_example(self, student, X, y, batch_size):
        return __generate_example__(self, student, X, y, batch_size)


class OmniscientConvTeacher(BaseConv):
    """
    Omnsicient teacher
    Pour un classifieur à convolution de classe OmniscientConvStudent
    """
    def select_example(self, student, X, y, batch_size):
        return __select_example__(self, student, X, y, batch_size)
