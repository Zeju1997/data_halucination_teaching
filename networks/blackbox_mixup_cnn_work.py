# https://github.com/jakeoung/BayesianUnrolling/blob/master/unroll/model/vanilla.py
# from . import attention
# from . import softargmax
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def mixup_data1(gt_x_1, gt_x_2, gt_y_1, gt_y_2, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    lam = alpha
    '''
    batch_size = gt_x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    '''
    y_a = gt_y_1
    y_b = gt_y_2
    mixed_x = lam * gt_x_1 + (1 - lam) * gt_x_2
    return mixed_x, y_a, y_b

def mixup_criterion1(criterion, pred, y_a, y_b, lam):
    loss = lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    return loss.to(torch.float32)

def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    batch_size = x.shape[0]
    index = torch.randperm(batch_size).cuda()
    if alpha > 0:
        lam = np.random.beta(alpha, alpha, size=(x.shape[0]))
        lam = torch.tensor(lam, dtype=torch.float).cuda()
        # mixed_y = lam * y + (1 - lam) * y[index]
        lam = torch.unsqueeze(lam, 1)
        lam = torch.unsqueeze(lam, 2)
        lam = torch.unsqueeze(lam, 3)
        mixed_x = lam * x + (1 - lam) * x[index, :]
    else:
        lam = 1
        mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    # mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    loss = lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    loss = torch.mean(loss)
    return loss

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.opt = opt
        self.label_embedding = nn.Embedding(self.opt.n_classes, self.opt.label_dim)
        self.img_shape = (self.opt.channels, self.opt.img_size, self.opt.img_size)
        in_channels = self.opt.label_dim + int(np.prod(self.img_shape))
        self.model = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, img, label):
        # Concatenate label embedding and image to produce input
        # d_in = torch.cat((img1.view(img1.size(0), -1), (img2.view(img2.size(0), -1), self.label_embedding(label1), self.label_embedding(label2)), -1))
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(label)), -1)
        validity = self.model(d_in)
        return validity

class Generator1(nn.Module):
    def __init__(self, opt, teacher, student):
        super(Generator, self).__init__()
        self.opt = opt
        self.label_emb = nn.Embedding(self.opt.n_classes, self.opt.label_dim)
        self.img_shape = (self.opt.channels, self.opt.img_size, self.opt.img_size)
        in_channels = int(np.prod(self.img_shape)) + self.opt.label_dim
        # in_channels = student.lin.weight.size(1) + self.opt.latent_dim * 2 + self.opt.label_dim * 2
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
            *block(512, 256),
            *block(256, 128),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, noise, label1, label2):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((noise, self.label_emb(label1.to(torch.int64)), self.label_emb(label2.to(torch.int64))), -1)
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
    def forward(self, img, label1, label2):
        # Concatenate label embedding and image to produce input
        # d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        d_in = torch.cat((img, self.label_embedding(label1), self.label_embedding(label2)), -1)
        validity = self.model(d_in)
        return validity

class Generator_moon(nn.Module):
    def __init__(self, opt, teacher, student):
        super(Generator_moon, self).__init__()
        self.opt = opt
        self.label_embedding = nn.Embedding(self.opt.n_classes, self.opt.label_dim)
        # in_channels = teacher.lin.weight.size(1) + student.lin.weight.size(1) + self.opt.dim + self.opt.label_dim
        in_channels = teacher.lin.weight.size(1) + self.opt.dim * 2 + self.opt.label_dim * 2
        self.input_fc = nn.Linear(in_channels, self.opt.hidden_dim*4, bias=False)
        self.hidden_fc = nn.Linear(self.opt.hidden_dim*4, self.opt.hidden_dim*2, bias=False)
        self.output_fc = nn.Linear(self.opt.hidden_dim*2, 1, bias=False)
        # self.activation = nn.LeakyReLU(0.1)
        self.activation = nn.ReLU()
        self.out_activation = nn.Sigmoid()

    def forward(self, z, label1, label2):
        x = torch.cat((z, self.label_embedding(label1.to(torch.int64)), self.label_embedding(label2.to(torch.int64))), dim=1)
        x = self.activation(self.input_fc(x))
        x = self.activation(self.hidden_fc(x))
        x = self.output_fc(x)
        x = self.out_activation(x)
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

class UnrolledBlackBoxOptimizer(nn.Module):
    """
    Args:
        - nscale : number of scales
        - alpha : scale factor in the softmax in the expansion (rho in the paper)
        - nblock : number of stages (K in the paper)
        - K : kernel size
    """
    def __init__(self, opt, teacher, student, generator, train_dataset, proj_matrix=None):
        super(UnrolledBlackBoxOptimizer, self).__init__()
        self.opt = opt
        self.optim_blocks = nn.ModuleList()
        # self.loss_fn = nn.MSELoss()
        self.loss_fn = nn.CrossEntropyLoss()
        self.adversarial_loss = nn.BCELoss()
        self.teacher = teacher
        self.student = student
        self.generator = generator
        self.dataset = train_dataset
        self.nb_batch = int(len(train_dataset) / self.opt.batch_size)
        self.proj_matrix = proj_matrix

    def forward(self, weight, w_star=None, w_init=None):
        # self.generator.linear.weight = weight
        # self.student.lin.weight = w_init
        optim_student = torch.optim.SGD(self.student.parameters(), lr=0.01)
        optim = torch.optim.SGD(self.generator.parameters(), lr=0.01)
        with torch.no_grad():
            # for param1 in self.generator.parameters():
            #    param1 = weight
            self.generator.load_state_dict(weight)
            self.student.load_state_dict(w_init)
            # for param1 in self.student.parameters():
            #     param1 = w_init
            # for param2 in self.teacher.parameters():
            #    param2 = w_star
        loss_stu = 0
        w_loss = 0
        tau = 1
        new_weight = w_init
        student_loss = []
        model_paramters = list(self.generator.parameters())
        student_parameters = list(self.student.parameters())

        self.student.train()
        for n in range(self.opt.n_unroll_blocks):
            # w_t = self.student.lin.weight

            # i = torch.randint(0, self.nb_batch, size=(1,)).item()
            # gt_x_1, gt_y_1 = self.data_sampler(self.X, self.Y, i)
            # i = torch.randint(0, self.nb_batch, size=(1,)).item()
            # gt_x_2, gt_y_2 = self.data_sampler(self.X, self.Y, i)
            i = torch.randint(0, self.nb_batch, size=(1,)).item()
            gt_x, gt_y = self.data_sampler(self.dataset, i)
            lam = self.generator(gt_x.cuda(), gt_y.cuda())
            batch_size = gt_x.shape[0]
            index = torch.randperm(batch_size).cuda()
            # lam = np.random.beta(alpha, alpha, size=(gt_x.shape[0]))
            # lam = torch.tensor(lam, dtype=torch.float).cuda()
            # mixed_y = lam * y + (1 - lam) * y[index]
            # lam = torch.unsqueeze(lam, 1)
            lam = torch.unsqueeze(lam, 2)
            lam = torch.unsqueeze(lam, 3)
            mixed_x = lam * gt_x + (1 - lam) * gt_x[index, :]
            targets_a, targets_b = gt_y, gt_y[index]
            # inputs, targets = inputs.cuda(), targets.long().cuda()
            # mixed_x, targets_a, targets_b, lam = mixup_data(gt_x, gt_y, alpha=1.0)
            outputs = self.student(mixed_x)
            loss = mixup_criterion(self.loss_fn, outputs, targets_a.long(), targets_b.long(), lam)
            # loss = self.loss_fn(outputs, mixed_y.long())
            grad_stu = torch.autograd.grad(outputs=loss,
                                           inputs=student_parameters,
                                           create_graph=True, retain_graph=True)
            with torch.no_grad():
                for param, grad in zip(self.student.parameters(), grad_stu):
                    new_param = param.data - 0.001 * grad
                    param.data = new_param

            loss_stu = loss_stu + loss # .clone()

            student_loss.append(loss.item())
            # print(n, "After backward pass", torch.cuda.memory_allocated(0))

            student_loss.append(loss.detach().item())

            '''
            mixup_baseline_optim.zero_grad()
            loss.backward()
            mixup_baseline_optim.step()
            # generated_y = torch.randint(0, 2, (self.opt.batch_size,), dtype=torch.float).cuda()
            # Sample noise and labels as generator input
            # z = Variable(torch.randn(gt_x.shape)).cuda()
            # z = Variable(torch.randn((self.opt.batch_size, self.opt.latent_dim))).cuda()
            # generated_x_proj = generated_x @ self.proj_matrix.cuda()
            # x = torch.cat((w_t, gt_x_1, gt_x_2), dim=1)
            # x = torch.cat((w_t, w_t-w_init, gt_x_1, gt_x_2), dim=1)
            # x = torch.cat((gt_x_1, gt_x_2), dim=0)
            alpha = self.generator(gt_x_1, gt_x_2, gt_y_1.long(), gt_y_2.long())
            # alpha = np.random.beta(1.0, 1.0)
            # alpha = torch.tensor(alpha, dtype=torch.float).cuda()
            #  alpha.requires_grad = True
            # mixup data
            # mixed_x, targets_a, targets_b = mixup_data(gt_x_1, gt_x_2, gt_y_1, gt_y_2, alpha)
            mixed_x = alpha * gt_x_1 + (1 - alpha) * gt_x_2
            # mixed_x, targets_a, targets_b = map(Variable, (mixed_x, targets_a, targets_b))
            # mixed_y = alpha * gt_y_1 + (1 - alpha) * gt_y_2
            # optim_student.zero_grad()
            # self.student.train()
            out = self.student(mixed_x)
            # out = self.student(generated_x_proj)
            # mixed_y = gt_y_1 * alpha + gt_y_2 * (1 - alpha)
            # loss = mixup_criterion(self.loss_fn, out, targets_a.float(), targets_b.float(), alpha)
            loss = alpha * self.loss_fn(out, gt_y_1) + (1 - alpha) * self.loss_fn(out, gt_y_2)
            loss = loss.to(torch.float32)
            # out_stu = gt_x
            # for idx, param in enumerate(self.student.parameters()):
            #     out_stu = param @ torch.transpose(out_stu, 0, 1)
            # out_stu = new_weight @ torch.transpose(gt_x, 0, 1)
            # out_stu = self.student(gt_x)
            # loss_stu = loss_stu + tau * self.loss_fn(out_stu, gt_y)
            # loss_stu = loss_stu + loss
            # grad = torch.autograd.grad(loss_stu, model_paramters, create_graph=True, retain_graph=True)
            # grad_stu = torch.autograd.grad(outputs=loss_stu, inputs=model_paramters, create_graph=True, retain_graph=True)
            optim.zero_grad()
            grad_stu = torch.autograd.grad(outputs=loss_stu, inputs=model_paramters, create_graph=True, retain_graph=True)
            loss_stu.backward()
            optim.step()
            '''
        grad_gen = torch.autograd.grad(outputs=loss_stu,
                                       inputs=model_paramters,
                                       create_graph=False, retain_graph=False)

        # w = self.generator.state_dict()

        # out_stu = new_weight @ torch.transpose(gt_x, 0, 1)
        # loss_stu = self.loss_fn(out_stu, gt_y)
        # alpha = 1
        # loss_stu = loss_stu / (self.opt.n_unroll_blocks * alpha)
        # loss_stu = loss_stu * alpha
        # grad_stu = torch.autograd.grad(outputs=loss_stu, inputs=model_paramters, create_graph=True, retain_graph=True)
        return grad_gen, loss_stu.item(), student_loss #, generated_x, gt_y, g_loss

    def data_sampler(self, dataset, i):
        i_min = i * self.opt.batch_size
        i_max = (i + 1) * self.opt.batch_size
        x = torch.tensor(dataset.data[i_min:i_max]).cuda()
        (N, W, H, C) = x.shape
        x = torch.reshape(x, (N, C, W, H))
        y = torch.tensor(dataset.targets[i_min:i_max]).cuda()
        return x, y
'''
def forward(self, weight, w_star=None, w_init=None):
        # self.generator.linear.weight = weight
        # self.student.lin.weight = w_init
        optim_student = torch.optim.SGD(self.student.parameters(), lr=0.01)
        optim = torch.optim.SGD(self.generator.parameters(), lr=0.01)
        with torch.no_grad():
            # for param1 in self.generator.parameters():
            #    param1 = weight
            self.generator.load_state_dict(weight)
            self.student.load_state_dict(w_init)
            # for param1 in self.student.parameters():
            #     param1 = w_init
            # for param2 in self.teacher.parameters():
            #    param2 = w_star
        loss_stu = 0
        w_loss = 0
        tau = 1
        new_weight = w_init
        student_loss = []
        model_paramters = list(self.generator.parameters())
        student_parameters = list(self.student.parameters())
        with torch.autograd.set_detect_anomaly(True):
            for i in range(self.opt.n_unroll_blocks):
                # w_t = self.student.lin.weight
                i = torch.randint(0, self.nb_batch, size=(1,)).item()
                gt_x_1, gt_y_1 = self.data_sampler(self.X, self.Y, i)
                i = torch.randint(0, self.nb_batch, size=(1,)).item()
                gt_x_2, gt_y_2 = self.data_sampler(self.X, self.Y, i)
                i = torch.randint(0, self.nb_batch, size=(1,)).item()
                i_min = i * self.opt.batch_size
                i_max = (i + 1) * self.opt.batch_size
                gt_x = self.X[i_min:i_max].cuda()
                gt_y = self.Y[i_min:i_max].cuda()
                # generated_y = torch.randint(0, 2, (self.opt.batch_size,), dtype=torch.float).cuda()
                # Sample noise and labels as generator input
                # z = Variable(torch.randn(gt_x.shape)).cuda()
                # z = Variable(torch.randn((self.opt.batch_size, self.opt.latent_dim))).cuda()
                # generated_x_proj = generated_x @ self.proj_matrix.cuda()
                # x = torch.cat((w_t, gt_x_1, gt_x_2), dim=1)
                # x = torch.cat((w_t, w_t-w_init, gt_x_1, gt_x_2), dim=1)
                x = torch.cat((gt_x_1, gt_x_2), dim=1)
                alpha = self.generator(x, gt_y_1.long(), gt_y_2.long())
                # alpha = np.random.beta(1.0, 1.0)
                # alpha = torch.tensor(alpha, dtype=torch.float).cuda()
                #  alpha.requires_grad = True
                # mixup data
                # mixed_x, targets_a, targets_b = mixup_data(gt_x_1, gt_x_2, gt_y_1, gt_y_2, alpha)
                mixed_x = alpha * gt_x_1 + (1 - alpha) * gt_x_2
                # mixed_x, targets_a, targets_b = map(Variable, (mixed_x, targets_a, targets_b))
                # mixed_y = alpha * gt_y_1 + (1 - alpha) * gt_y_2
                optim_student.zero_grad()
                # self.student.train()
                out = self.student(mixed_x)
                # out = self.student(generated_x_proj)
                # mixed_y = gt_y_1 * alpha + gt_y_2 * (1 - alpha)
                # loss = mixup_criterion(self.loss_fn, out, targets_a.float(), targets_b.float(), alpha)
                loss = alpha.clone() * self.loss_fn(out.clone(), gt_y_1) + (1 - alpha.clone()) * self.loss_fn(out, gt_y_2)
                loss = loss.to(torch.float32)
                # grad = torch.autograd.grad(loss, alpha, create_graph=True, retain_graph=True)
                # grad = torch.autograd.grad(alpha, model_paramters, create_graph=True, retain_graph=True)
                # grad = torch.autograd.grad(loss, model_paramters, create_graph=True, retain_graph=True)
                loss_stu = loss_stu + loss.clone()
                loss.backward(create_graph=True, retain_graph=True)
                optim_student.step()
                # grad = torch.autograd.grad(loss, student_parameters, create_graph=True, retain_graph=True)
                # new_weight = self.student.lin.weight - 0.001 * grad[0]
                # for idx, param in enumerate(self.student.parameters()):
                #    param = param - 0.001 * grad[idx]
                # new_weight = new_weight - 0.001 * grad[0]
                # self.student.lin.weight = torch.nn.Parameter(new_weight.cuda())
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
                 # i = torch.randint(0, nb_batch, size=(1,)).item()
                # gt_x_1, gt_y_1 = self.data_sampler(X_train, Y_train, i)
                # i = torch.randint(0, nb_batch, size=(1,)).item()
                # gt_x_2, gt_y_2 = self.data_sampler(X_train, Y_train, i)
                # x = torch.cat((gt_x_1, gt_x_2), dim=1)
                # alpha = self.generator(x, gt_y_1.long(), gt_y_2.long())
                # mixed_x = alpha * gt_x_1 + (1 - alpha) * gt_x_2
                # optim.zero_grad()
                # out = self.student(mixed_x)
                # loss = mixup_criterion(self.loss_fn, out, gt_y_1.float(), gt_y_2.float(), alpha)
                # loss = alpha * self.loss_fn(out, gt_y_1.long()) + (1 - alpha) * self.loss_fn(out, gt_y_2.long())
                # loss = loss.to(torch.float32)
                # loss.backward()
                # optim.step()
                # loss_stu = loss_stu + tau * self.loss_fn(out_stu, gt_y)
                # loss_stu = loss_stu + tau * mixup_criterion(self.loss_fn, out_stu, targets_a.float(), targets_b.float(), lam)
                student_loss.append(loss.item())
                # out_stu = gt_x
                # for idx, param in enumerate(self.student.parameters()):
                #     out_stu = param @ torch.transpose(out_stu, 0, 1)
                # out_stu = new_weight @ torch.transpose(gt_x, 0, 1)
                # out_stu = self.student(gt_x)
                # loss_stu = loss_stu + tau * self.loss_fn(out_stu, gt_y)
                # loss_stu = loss_stu + loss
                # grad = torch.autograd.grad(loss_stu, model_paramters, create_graph=True, retain_graph=True)
                # grad_stu = torch.autograd.grad(outputs=loss_stu, inputs=model_paramters, create_graph=True, retain_graph=True)
                optim.zero_grad()
                grad_stu = torch.autograd.grad(outputs=loss_stu, inputs=model_paramters, create_graph=True, retain_graph=True)
                loss_stu.backward()
                optim.step()
            w = self.generator.state_dict()
            # out_stu = new_weight @ torch.transpose(gt_x, 0, 1)
            # loss_stu = self.loss_fn(out_stu, gt_y)
            # alpha = 1
            # loss_stu = loss_stu / (self.opt.n_unroll_blocks * alpha)
            # loss_stu = loss_stu * alpha
            # grad_stu = torch.autograd.grad(outputs=loss_stu, inputs=model_paramters, create_graph=True, retain_graph=True)
        return w, loss_stu, student_loss #, generated_x, gt_y, g_loss
'''
