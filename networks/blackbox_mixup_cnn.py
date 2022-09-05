# https://github.com/jakeoung/BayesianUnrolling/blob/master/unroll/model/vanilla.py

# from . import attention
# from . import softargmax

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import one_hot, log_softmax, softmax, normalize
from torchvision import transforms
from torchvision.models import resnet18

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

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])

class Generator2(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.affine1 = nn.Linear(3, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 1)

        self.act = nn.Sigmoid()

        self.saved_log_probs = []
        self.rewards = []

        self.opt = opt

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return self.act(action_scores) * 0.5

class Generator(nn.Module):
    def __init__(self, opt, feature_extractor=None):
        super(Generator, self).__init__()

        self.opt = opt
        self.label_embedding = nn.Embedding(self.opt.n_classes, self.opt.label_dim)
        self.img_shape = (self.opt.channels, self.opt.img_size, self.opt.img_size)

        # self.backbone = resnet18(pretrained=True)
        # self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # self.backbone.fc = nn.Identity()
        # self.backbone.requires_grad_(False)

        in_channels = self.opt.label_dim*2 + self.opt.n_classes*2

        self.model = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 128),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 128),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 10),
            # nn.Sigmoid()
        )
        # feat_dim = torch.combinations(torch.arange(self.opt.n_query_classes))
        # feat_dim = self.opt.n_query_classes

        self.fc1 = nn.Linear(self.opt.n_classes + 3, 5)

        self.act = nn.Sigmoid()

    def forward(self, img1, img2, label1, label2, feat_model):
        # feat_1 = self.backbone(img1)
        # feat_2 = self.backbone(img2)
        # Concatenate label embedding and image to produce input
        # d_in = torch.cat((img1.view(img1.size(0), -1), img2.view(img2.size(0), -1), self.label_embedding(label1), self.label_embedding(label2)), -1)
        d_in = torch.cat((img1, img2, self.label_embedding(label1), self.label_embedding(label2)), -1)
        x = self.model(d_in)

        # feat_sim = torch.tensor(feat_sim).unsqueeze(0).repeat(img.shape[0], 1)
        feat_model = feat_model.unsqueeze(0).repeat(x.shape[0], 1)
        x = torch.cat((x, feat_model), dim=1)
        x = self.act(self.fc1(x))

        x = F.softmax(x, dim=1)

        out = x[:, 1] * 0.2 + x[:, 2] * 0.4 + x[:, 3] * 0.6 + x[:, 4] * 0.8

        return out


class Generator1(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()

        self.opt = opt
        self.label_embedding = nn.Embedding(self.opt.n_classes, self.opt.label_dim)
        self.img_shape = (self.opt.channels, self.opt.img_size, self.opt.img_size)

        self.offset = torch.tensor([-0.1, 0, 0.1]).cuda()

        in_channels = self.opt.label_dim + int(np.prod(self.img_shape))

        self.model = nn.Sequential(
            nn.Linear(in_channels, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 256),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, self.opt.n_classes),
            # nn.Sigmoid()
        )
        # feat_dim = torch.combinations(torch.arange(self.opt.n_query_classes))
        # feat_dim = self.opt.n_query_classes

        # self.fc1 = nn.Linear(self.opt.n_classes + feat_dim, 16)
        # self.fc2 = nn.Linear(16, 10)
        self.fc = nn.Linear(10 + 4, 3)

        # self.act = nn.Sigmoid()
        self.act = nn.Softmax(dim=1)

    def forward(self, img, label, feat_model, lam):

        # Concatenate label embedding and image to produce input
        # d_in = torch.cat((img1.view(img1.size(0), -1), (img2.view(img2.size(0), -1), self.label_embedding(label1), self.label_embedding(label2)), -1))
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(label)), -1)
        x = self.model(d_in)

        # feat = feat_sim.unsqueeze(0).repeat(img.shape[0], 1)
        # x = torch.cat((x, feat), dim=1)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))

        feat_model = feat_model.unsqueeze(0).repeat(x.shape[0], 1)
        lam_repeat = lam.unsqueeze(0).repeat(x.shape[0], 1)
        x = torch.cat((x, feat_model, lam_repeat), dim=1)
        # actions = self.fc3(x)
        x = self.act(self.fc(x))

        offset = x @ self.offset

        print("offset", offset.mean())

        lam = lam + offset.mean()
        val_lam = torch.clamp(lam, min=0, max=1)

        return val_lam


class Generator2(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()

        self.opt = opt
        self.label_embedding = nn.Embedding(self.opt.n_classes, self.opt.label_dim)
        self.img_shape = (self.opt.channels, self.opt.img_size, self.opt.img_size)

        in_channels = self.opt.label_dim + int(np.prod(self.img_shape))

        self.model = nn.Sequential(
            nn.Linear(in_channels, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 256),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, self.opt.n_classes),
            # nn.Sigmoid()
        )
        # feat_dim = torch.combinations(torch.arange(self.opt.n_query_classes))
        feat_dim = self.opt.n_query_classes

        self.fc1 = nn.Linear(self.opt.n_classes + feat_dim, 32)
        self.fc2 = nn.Linear(32, 10)
        self.fc3 = nn.Linear(10 + 3, 1)

        self.act = nn.Sigmoid()

    def forward(self, img, label, feat_model, feat_sim):
        # Concatenate label embedding and image to produce input
        # d_in = torch.cat((img1.view(img1.size(0), -1), (img2.view(img2.size(0), -1), self.label_embedding(label1), self.label_embedding(label2)), -1))
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(label)), -1)
        x = self.model(d_in)

        feat = feat_sim.unsqueeze(0).repeat(img.shape[0], 1)
        x = torch.cat((x, feat), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        feat_model = feat_model.unsqueeze(0).repeat(x.shape[0], 1)
        x = torch.cat((x, feat_model), dim=1)
        x = self.act(self.fc3(x))

        return x


class Generator1(nn.Module):
    def __init__(self, opt):
        super(Generator1, self).__init__()
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


class UnrolledBlackBoxOptimizer1(nn.Module):
    """
    Args:
        - nscale : number of scales
        - alpha : scale factor in the softmax in the expansion (rho in the paper)
        - nblock : number of stages (K in the paper)
        - K : kernel size
    """
    def __init__(self, opt, teacher, student, generator, train_dataset, val_loader, proj_matrix=None):
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

        self.netG_optim = torch.optim.Adam(self.generator.model.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.001)

        self.student_optim = torch.optim.SGD(self.student.parameters(), lr=0.001, momentum=0.9, weight_decay=self.opt.decay)

        self.val_loader = val_loader

    def forward(self, weight, model_features):
        # self.generator.linear.weight = weight
        # self.student.lin.weight = w_init
        optim_student = torch.optim.SGD(self.student.parameters(), lr=0.01)
        optim = torch.optim.SGD(self.generator.parameters(), lr=0.01)
        with torch.no_grad():
            # for param1 in self.generator.parameters():
            #    param1 = weight
            self.generator.load_state_dict(weight)
            # self.student.load_state_dict(w_init)
            # for param1 in self.student.parameters():
            #     param1 = w_init
            # for param2 in self.teacher.parameters():
            #    param2 = w_star
        loss_stu = 0
        student_loss = []
        model_paramters = list(self.generator.model.parameters())
        student_parameters = list(self.student.parameters())

        loader_eval = iter(self.val_loader)

        self.student.train()
        for n in range(self.opt.n_unroll_blocks):
            self.student.eval()

            try:
                (val_inputs, val_targets) = loader_eval.next()
            except:
                loader_eval = iter(self.val_loader)
                (val_inputs, val_targets) = loader_eval.next()

            val_inputs, val_targets = val_inputs.cuda(), val_targets.long().cuda()

            val_targets_onehot = one_hot(val_targets, self.opt.n_classes)

            self.generator.train()

            # val_lam = np.random.beta(1.0, 1.0)
            # val_lam = torch.tensor(val_lam, dtype=torch.float).cuda()
            # val_offset = netG(val_inputs.cuda(), val_targets.cuda(), model_features, val_lam)
            # val_offset = torch.max(val_offset, dim=1).indices
            # offset_count = torch.bincount(val_offset)

            # if torch.argmax(offset_count) == 0:
            #     offset = 0.1
            # elif torch.argmax(offset_count) == 2:
            #     offset = - 0.1
            # else:
            #     offset = 0

            # val_lam = val_lam + offset
            # val_lam = torch.clamp(val_lam, min=0, max=1)

            index = torch.randperm(val_inputs.shape[0]).cuda()

            val_inputs_logits_1 = self.student(val_inputs)
            val_inputs_logits_2 = self.student(val_inputs[index, :])
            val_lam = self.generator(val_inputs_logits_1, val_inputs_logits_2, val_targets, val_targets[index], model_features)
            ## val_lam = self.generator(val_inputs, val_inputs[index, :], val_targets, val_targets[index], model_features)

            x_lam = torch.reshape(val_lam, (val_inputs.shape[0], 1, 1, 1)).cuda()
            y_lam = torch.reshape(val_lam, (val_inputs.shape[0], 1)).cuda()

            mixed_x = x_lam * val_inputs + (1 - x_lam) * val_inputs[index, :]
            mixed_y = y_lam * val_targets_onehot + (1 - y_lam) * val_targets_onehot[index]

            outputs_mixed = self.student(mixed_x)

            # loss = val_lam * self.loss_fn(val_outputs, targets_a) + (1 - val_lam) * self.loss_fn(val_outputs, targets_b)

            loss = self.loss_fn(outputs_mixed, mixed_y)

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

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        outputs = self.student(input)
        loss = self.loss_fn(outputs, target)
        self.student_optim.zero_grad()
        loss.backward(retain_graph=True)
        self.student_optim.step()

        # dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta

        '''
        theta = _concat(self.model.parameters()).data
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta
        unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
        '''

        # return self.student

    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, model_features):
        targets_onehot = one_hot(target_train, self.opt.n_classes)

        index = torch.randperm(input_train.shape[0]).cuda()

        inputs_logits_1 = self.student(input_train)
        inputs_logits_2 = self.student(input_train[index, :])
        lam = self.generator(inputs_logits_1, inputs_logits_2, target_train, target_train[index], model_features)
        # lam = self.generator(input_train, input_train[index, :], target_train, target_train[index], model_features)
        # lam = self.generator(model_features)

        x_lam = torch.reshape(lam, (input_train.shape[0], 1, 1, 1)).cuda()
        # x_lam = lam
        y_lam = torch.reshape(lam, (input_train.shape[0], 1)).cuda()
        # y_lam = lam

        mixed_x = x_lam * input_train + (1 - x_lam) * input_train[index, :]
        mixed_y = y_lam * targets_onehot + (1 - y_lam) * targets_onehot[index]

        self.netG_optim.zero_grad()
        unrolled_loss = self._backward_step_unrolled(mixed_x, mixed_y, input_valid, target_valid, eta, network_optimizer, model_features)
        self.netG_optim.step()
        return unrolled_loss

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, model_features):
        self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)

        self.student.zero_grad()
        self.generator.zero_grad()

        target_valid_onehot = one_hot(target_valid, self.opt.n_classes)

        index = torch.randperm(input_valid.shape[0]).cuda()

        val_inputs_logits_1 = self.student(input_valid)
        val_inputs_logits_2 = self.student(input_valid[index, :])
        val_lam = self.generator(val_inputs_logits_1, val_inputs_logits_2, target_valid, target_valid[index], model_features)
        # val_lam = self.generator(input_valid, input_valid[index, :], target_valid, target_valid[index], model_features)
        # val_lam = self.generator(model_features)

        x_lam = torch.reshape(val_lam, (input_valid.shape[0], 1, 1, 1)).cuda()
        # x_lam = val_lam
        y_lam = torch.reshape(val_lam, (input_valid.shape[0], 1)).cuda()
        # y_lam = val_lam

        mixed_x = x_lam * input_valid + (1 - x_lam) * input_valid[index, :]
        mixed_y = y_lam * target_valid_onehot + (1 - y_lam) * target_valid_onehot[index]

        outputs_mixed = self.student(mixed_x)
        unrolled_loss = self.loss_fn(outputs_mixed, mixed_y)

        # output_valid = self.student(input_valid)
        # unrolled_loss = self.loss_fn(output_valid, target_valid)

        unrolled_loss.backward()
        dalpha = [v.grad for v in self.generator.model.parameters()]
        vector = [v.grad.data for v in self.student.parameters()]
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train, model_features)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        for v, g in zip(self.generator.model.parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

        return unrolled_loss.item()

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset+v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, inputs, targets, model_features, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.student.parameters(), vector):
            p.data.add_(R, v)
        outputs = self.student(inputs)
        loss = self.loss_fn(outputs, targets)
        grads_p = torch.autograd.grad(loss, self.generator.model.parameters(), retain_graph=True)
        # self.generator.zero_grad()

        for p, v in zip(self.student.parameters(), vector):
            p.data.sub_(2*R, v)
        outputs = self.student(inputs)
        loss = self.loss_fn(outputs, targets)
        grads_n = torch.autograd.grad(loss, self.generator.model.parameters(), retain_graph=True)
        # self.generator.zero_grad()

        for p, v in zip(self.student.parameters(), vector):
            p.data.add_(R, v)

        return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

    def data_sampler(self, dataset, i):
        i_min = i * self.opt.batch_size
        i_max = (i + 1) * self.opt.batch_size

        x = torch.tensor(dataset.data[i_min:i_max]).cuda()
        (N, W, H, C) = x.shape
        x = torch.reshape(x, (N, C, W, H))
        y = torch.tensor(dataset.targets[i_min:i_max]).cuda()

        return x, y

    def model_features(self, train_loss, epoch):
        current_iter = self.step / (self.opt.n_epochs * len(self.train_loader))

        if epoch == 1:
            avg_training_loss = 1.0
        else:
            avg_training_loss = train_loss / self.init_train_loss

        if epoch == 1:
            best_val_loss = 1.0
        else:
            best_val_loss = self.best_test_loss / self.init_test_loss
        model_features = [current_iter, avg_training_loss, best_val_loss]
        return torch.FloatTensor(model_features).cuda()


class UnrolledBlackBoxOptimizer(nn.Module):
    """
    Args:
        - nscale : number of scales
        - alpha : scale factor in the softmax in the expansion (rho in the paper)
        - nblock : number of stages (K in the paper)
        - K : kernel size
    """
    def __init__(self, opt, teacher, student, generator, train_dataset, val_loader, proj_matrix=None):
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

        self.netG_optim = torch.optim.Adam(self.generator.model.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.001)

        self.student_optim = torch.optim.SGD(self.student.parameters(), lr=0.001, momentum=0.9, weight_decay=self.opt.decay)

        self.val_loader = val_loader

    def forward(self, weight, model_features):
        # self.generator.linear.weight = weight
        # self.student.lin.weight = w_init
        optim_student = torch.optim.SGD(self.student.parameters(), lr=0.01)
        optim = torch.optim.SGD(self.generator.parameters(), lr=0.01)
        # with torch.no_grad():
            # for param1 in self.generator.parameters():
            #    param1 = weight
            # self.generator.load_state_dict(weight)
            # self.student.load_state_dict(w_init)
            # for param1 in self.student.parameters():
            #     param1 = w_init
            # for param2 in self.teacher.parameters():
            #    param2 = w_star
        loss_stu = 0
        student_loss = []
        model_paramters = list(self.generator.model.parameters())
        student_parameters = list(self.student.parameters())

        loader_eval = iter(self.val_loader)

        self.student.train()
        for n in range(self.opt.n_unroll_blocks):
            self.student.eval()

            try:
                (val_inputs, val_targets) = loader_eval.next()
            except:
                loader_eval = iter(self.val_loader)
                (val_inputs, val_targets) = loader_eval.next()

            val_inputs, val_targets = val_inputs.cuda(), val_targets.long().cuda()

            val_targets_onehot = one_hot(val_targets, self.opt.n_classes)

            self.generator.train()

            # val_lam = np.random.beta(1.0, 1.0)
            # val_lam = torch.tensor(val_lam, dtype=torch.float).cuda()
            # val_offset = netG(val_inputs.cuda(), val_targets.cuda(), model_features, val_lam)
            # val_offset = torch.max(val_offset, dim=1).indices
            # offset_count = torch.bincount(val_offset)

            # if torch.argmax(offset_count) == 0:
            #     offset = 0.1
            # elif torch.argmax(offset_count) == 2:
            #     offset = - 0.1
            # else:
            #     offset = 0

            # val_lam = val_lam + offset
            # val_lam = torch.clamp(val_lam, min=0, max=1)

            index = torch.randperm(val_inputs.shape[0]).cuda()

            val_inputs_logits_1 = self.student(val_inputs)
            val_inputs_logits_2 = self.student(val_inputs[index, :])
            val_lam = self.generator(val_inputs_logits_1, val_inputs_logits_2, val_targets, val_targets[index], model_features)
            ## val_lam = self.generator(val_inputs, val_inputs[index, :], val_targets, val_targets[index], model_features)

            x_lam = torch.reshape(val_lam, (val_inputs.shape[0], 1, 1, 1)).cuda()
            y_lam = torch.reshape(val_lam, (val_inputs.shape[0], 1)).cuda()

            mixed_x = x_lam * val_inputs + (1 - x_lam) * val_inputs[index, :]
            mixed_y = y_lam * val_targets_onehot + (1 - y_lam) * val_targets_onehot[index]

            outputs_mixed = self.student(mixed_x)

            # loss = val_lam * self.loss_fn(val_outputs, targets_a) + (1 - val_lam) * self.loss_fn(val_outputs, targets_b)

            loss = self.loss_fn(outputs_mixed, mixed_y)

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
        grad_gen = torch.autograd.grad(outputs=loss_stu,
                                       inputs=model_paramters,
                                       create_graph=False, retain_graph=False)
        '''
        grad_gen = 0

        return grad_gen, loss_stu.item(), loss_stu #, generated_x, gt_y, g_loss

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        outputs = self.student(input)
        loss = self.loss_fn(outputs, target)
        self.student_optim.zero_grad()
        loss.backward(retain_graph=True)
        self.student_optim.step()

        # dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta

        '''
        theta = _concat(self.model.parameters()).data
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta
        unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
        '''

        # return self.student

    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, model_features):
        targets_onehot = one_hot(target_train, self.opt.n_classes)

        index = torch.randperm(input_train.shape[0]).cuda()

        inputs_logits_1 = self.student(input_train)
        inputs_logits_2 = self.student(input_train[index, :])
        lam = self.generator(inputs_logits_1, inputs_logits_2, target_train, target_train[index], model_features)
        # lam = self.generator(input_train, input_train[index, :], target_train, target_train[index], model_features)
        # lam = self.generator(model_features)

        x_lam = torch.reshape(lam, (input_train.shape[0], 1, 1, 1)).cuda()
        # x_lam = lam
        y_lam = torch.reshape(lam, (input_train.shape[0], 1)).cuda()
        # y_lam = lam

        mixed_x = x_lam * input_train + (1 - x_lam) * input_train[index, :]
        mixed_y = y_lam * targets_onehot + (1 - y_lam) * targets_onehot[index]

        self.netG_optim.zero_grad()
        unrolled_loss = self._backward_step_unrolled(mixed_x, mixed_y, input_valid, target_valid, eta, network_optimizer, model_features)
        self.netG_optim.step()
        return unrolled_loss

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, model_features):
        self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)

        # w_t = self.student.state_dict()
        w_t = self.generator.state_dict()
        gradients, generator_loss, unrolled_loss = self.forward(w_t, model_features)

        self.student.train()

        # with torch.no_grad():
        #     for p, g in zip(self.generator.model.parameters(), gradients):
        #        p.grad = g

        # optimizer_G.step()
        # netG.eval()

        '''
        target_valid_onehot = one_hot(target_valid, self.opt.n_classes)

        index = torch.randperm(input_valid.shape[0]).cuda()

        val_inputs_logits_1 = self.student(input_valid)
        val_inputs_logits_2 = self.student(input_valid[index, :])
        val_lam = self.generator(val_inputs_logits_1, val_inputs_logits_2, target_valid, target_valid[index], model_features)
        # val_lam = self.generator(input_valid, input_valid[index, :], target_valid, target_valid[index], model_features)
        # val_lam = self.generator(model_features)

        x_lam = torch.reshape(val_lam, (input_valid.shape[0], 1, 1, 1)).cuda()
        # x_lam = val_lam
        y_lam = torch.reshape(val_lam, (input_valid.shape[0], 1)).cuda()
        # y_lam = val_lam

        mixed_x = x_lam * input_valid + (1 - x_lam) * input_valid[index, :]
        mixed_y = y_lam * target_valid_onehot + (1 - y_lam) * target_valid_onehot[index]

        outputs_mixed = self.student(mixed_x)
        unrolled_loss = self.loss_fn(outputs_mixed, mixed_y)

        # output_valid = self.student(input_valid)
        # unrolled_loss = self.loss_fn(output_valid, target_valid)
        '''

        unrolled_loss.backward()
        dalpha = [v.grad for v in self.generator.model.parameters()]
        vector = [v.grad.data for v in self.student.parameters()]
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train, model_features)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        for v, g in zip(self.generator.model.parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

        return unrolled_loss.item()

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset+v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, inputs, targets, model_features, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.student.parameters(), vector):
            p.data.add_(R, v)
        outputs = self.student(inputs)
        loss = self.loss_fn(outputs, targets)
        grads_p = torch.autograd.grad(loss, self.generator.model.parameters(), retain_graph=True)
        # self.generator.zero_grad()

        for p, v in zip(self.student.parameters(), vector):
            p.data.sub_(2*R, v)
        outputs = self.student(inputs)
        loss = self.loss_fn(outputs, targets)
        grads_n = torch.autograd.grad(loss, self.generator.model.parameters(), retain_graph=True)
        # self.generator.zero_grad()

        for p, v in zip(self.student.parameters(), vector):
            p.data.add_(R, v)

        return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

    def data_sampler(self, dataset, i):
        i_min = i * self.opt.batch_size
        i_max = (i + 1) * self.opt.batch_size

        x = torch.tensor(dataset.data[i_min:i_max]).cuda()
        (N, W, H, C) = x.shape
        x = torch.reshape(x, (N, C, W, H))
        y = torch.tensor(dataset.targets[i_min:i_max]).cuda()

        return x, y

    def model_features(self, train_loss, epoch):
        current_iter = self.step / (self.opt.n_epochs * len(self.train_loader))

        if epoch == 1:
            avg_training_loss = 1.0
        else:
            avg_training_loss = train_loss / self.init_train_loss

        if epoch == 1:
            best_val_loss = 1.0
        else:
            best_val_loss = self.best_test_loss / self.init_test_loss
        model_features = [current_iter, avg_training_loss, best_val_loss]
        return torch.FloatTensor(model_features).cuda()

