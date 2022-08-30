import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn.functional import one_hot, log_softmax, softmax, normalize
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from .controller import Agent, Policy
from collections import deque
# from model import NASModel
import networks

from tqdm import tqdm
from itertools import count

activation = {}

class PolicyGradient:
    def __init__(self, opt, student, train_loader, val_loader, test_loader, writers):
        NUM_EPOCHS = 50
        ALPHA = 5e-3        # learning rate
        BATCH_SIZE = 1     # how many episodes we want to pack into an epoch
        HIDDEN_SIZE = 64    # number of hidden nodes we have in our dnn
        BETA = 0.1          # the entropy bonus multiplier
        INPUT_SIZE = 3
        ACTION_SPACE = 11
        NUM_STEPS = 4
        GAMMA = 0.99

        self.experiment = "policy gradient"

        self.student = student
        self.student.lin2.register_forward_hook(self.get_activation('latent'))

        self.NUM_EPOCHS = NUM_EPOCHS
        self.ALPHA = ALPHA
        self.BATCH_SIZE = BATCH_SIZE # number of models to generate for each action
        self.HIDDEN_SIZE = HIDDEN_SIZE
        self.BETA = BETA
        self.GAMMA = GAMMA
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.INPUT_SIZE = INPUT_SIZE
        self.NUM_STEPS = NUM_STEPS
        self.ACTION_SPACE = ACTION_SPACE

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.opt = opt

        # instantiate the tensorboard writer
        """
        self.writer = SummaryWriter(comment=f'_PG_CP_Gamma={self.GAMMA},'
                                            f'LR={self.ALPHA},'
                                            f'BS={self.BATCH_SIZE},'
                                            f'NH={self.HIDDEN_SIZE},'
                                            f'BETA={self.BETA}')
        """
        self.writers = writers

        # the agent driven by a neural network architecture
        # self.agent = Agent(self.opt, self.INPUT_SIZE, self.HIDDEN_SIZE, self.NUM_STEPS, device=self.DEVICE).cuda()
        self.agent = Policy().cuda()

        self.adam = optim.Adam(params=self.agent.parameters(), lr=self.ALPHA)
        self.total_rewards = deque([], maxlen=100)

        print(self.agent)

        self.step = 0
        self.best_acc = 0
        self.best_test_loss = 0
        self.init_train_loss = 0
        self.init_test_loss = 0
        self.init_feat_sim = 0
        self.query_set_1, self.query_set_2 = self.get_query_set()

    def get_activation(self, name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    def get_query_set(self):
        """decrease the learning rate at 100 and 150 epoch"""
        query_set_1 = torch.empty(self.opt.n_query_classes, self.opt.channels, self.opt.img_size, self.opt.img_size)
        query_set_2 = torch.empty(self.opt.n_query_classes, self.opt.channels, self.opt.img_size, self.opt.img_size)
        val_iter = iter(self.val_loader)
        for i in tqdm(range(self.opt.n_classes)):
            while True:
                try:
                    (inputs, targets) = val_iter.next()
                except:
                    val_iter = iter(self.val_loader)
                    (inputs, targets) = val_iter.next()
                idx = ((targets == i).nonzero(as_tuple=True)[0])
                if idx.nelement() == 0:
                    pass
                else:
                    idx = idx[0]
                    query_set_1[i, :] = inputs[idx, :]
                    break

        for i in tqdm(range(self.opt.n_classes)):
            while True:
                try:
                    (inputs, targets) = val_iter.next()
                except:
                    val_iter = iter(self.val_loader)
                    (inputs, targets) = val_iter.next()
                idx = ((targets == i).nonzero(as_tuple=True)[0])
                if idx.nelement() == 0:
                    pass
                else:
                    idx = idx[0]
                    query_set_2[i, :] = inputs[idx, :]
                    break

        return query_set_1.cuda(), query_set_2.cuda()

    def select_action1(self, policy, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = policy(state)
        m = Categorical(probs)
        action = m.sample()
        policy.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def select_action(self, state):
        # get the action logits from the agent - (preferences)
        episode_logits = self.agent(torch.tensor(state).float().to(self.DEVICE))

        # sample an action according to the action distribution
        action_index = Categorical(logits=episode_logits).sample().unsqueeze(1)

        mask = one_hot(action_index, num_classes=self.ACTION_SPACE)

        episode_log_probs = torch.sum(mask.float() * log_softmax(episode_logits, dim=1), dim=1)

        # append the action to the episode action list to obtain the trajectory
        # we need to store the actions and logits so we could calculate the gradient of the performance
        #episode_actions = torch.cat((episode_actions, action_index), dim=0)

        # Get action actions
        action_space = torch.tensor([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], device=self.DEVICE).unsqueeze(0).repeat(self.opt.n_epochs, 1)
        # action_space = torch.tensor([[0.2, 0.3, 1.0], [0.2, 0.3, 1.0], [0.2, 0.3, 1.0], [0.2, 0.3, 1.0]], device=self.DEVICE)

        action = torch.gather(action_space, 1, action_index).squeeze(1)
        return action.item(), episode_logits, episode_log_probs

    def finish_episode(self, policy):
        R = 0
        policy_loss = []
        returns = []
        for r in policy.rewards[::-1]:
            R = r + args.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        for log_prob, R in zip(policy.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()
        del policy.rewards[:]
        del policy.saved_log_probs[:]

    def query_model(self):
        _ = self.student(self.query_set_1)
        act1 = activation['latent'].squeeze()
        _ = self.student(self.query_set_2)
        act2 = activation['latent'].squeeze()
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        feat_sim = cos(act1, act2)

        return feat_sim.cuda()

    def model_features(self, train_loss):
        current_iter = self.step / (self.opt.n_epochs * len(self.train_loader))

        avg_training_loss = train_loss / self.init_train_loss

        best_val_loss = self.best_test_loss / self.init_test_loss

        model_features = [[current_iter, avg_training_loss, best_val_loss]]
        return model_features
        # return torch.FloatTensor(model_features).cuda()

    def avg_loss(self, model, data_loader, n_iter):
        model.eval()
        train_loss = 0
        correct = 0
        loss_fn = nn.CrossEntropyLoss(reduction='sum')
        train_iter = iter(data_loader)
        with torch.no_grad():
            for i in range(n_iter):
                try:
                    (inputs, targets) = train_iter.next()
                except:
                    train_iter = iter(data_loader)
                    (inputs, targets) = train_iter.next()

                inputs, targets = inputs.cuda(), targets.cuda()
                output = model(inputs)

                train_loss += loss_fn(output, targets.long()).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(targets.view_as(pred)).sum().item()

        train_loss /= n_iter

        return train_loss

    def solve_environment(self):
        """
            The main interface for the Policy Gradient solver
        """

        # init the episode and the epoch
        epoch = 0

        while epoch < self.NUM_EPOCHS:
            # init the epoch arrays
            # used for entropy calculation
            epoch_logits = torch.empty(size=(0, self.ACTION_SPACE), device=self.DEVICE)
            epoch_weighted_log_probs = torch.empty(size=(0,), dtype=torch.float, device=self.DEVICE)

            # Sample BATCH_SIZE models and do average
            for i in range(self.BATCH_SIZE):
                # play an episode of the environment
                (episode_weighted_log_prob_trajectory,
                 episode_logits,
                 sum_of_episode_rewards) = self.play_episode()

                # after each episode append the sum of total rewards to the deque
                self.total_rewards.append(sum_of_episode_rewards)

                # append the weighted log-probabilities of actions
                epoch_weighted_log_probs = torch.cat((epoch_weighted_log_probs, episode_weighted_log_prob_trajectory), dim=0)

                # append the logits - needed for the entropy bonus calculation
                epoch_logits = torch.cat((epoch_logits, episode_logits), dim=0)

            # calculate the loss
            loss, entropy = self.calculate_loss(epoch_logits=epoch_logits,
                                                weighted_log_probs=epoch_weighted_log_probs)

            # zero the gradient
            self.adam.zero_grad()

            # backprop
            loss.backward()

            # update the parameters
            self.adam.step()

            # feedback
            print("\r", f"Epoch: {epoch}, Avg Return per Epoch: {np.mean(self.total_rewards):.3f}",
                  end="",
                  flush=True)

            self.log(mode="policy gradient", name="Average Return over 100 episodes", value=np.mean(self.total_rewards), step=epoch)
            self.log(mode="policy gradient", name="Entropy", value=entropy, step=epoch)

            # check if solved
            # if np.mean(self.total_rewards) > 200:
            #     print('\nSolved!')
            #     break
            epoch += 1
        # close the writer
        self.writer.close()

    def play_episode(self):
        """
            Plays an episode of the environment.
            episode: the episode counter
            Returns:
                sum_weighted_log_probs: the sum of the log-prob of an action multiplied by the reward-to-go from that state
                episode_logits: the logits of every step of the episode - needed to compute entropy for entropy bonus
                finished_rendering_this_epoch: pass-through rendering flag
                sum_of_rewards: sum of the rewards for the episode - needed for the average over 200 episode statistic
        """

        self.step = 0
        self.best_acc = 0
        self.best_test_loss = 1000
        self.init_train_loss = 0
        self.init_test_loss = 0
        self.init_feat_sim = 0
        self.test_loss = 0

        # Init state
        state = [[0, 1.0, 1.0]]

        # generate a submodel given predicted actions
        # net = NASModel(action)
        net = networks.CNN(in_channels=self.opt.channels, num_classes=self.opt.n_classes).cuda()
        #net = Net()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        running_loss = 0.0
        avg_train_loss = 0.0
        train_loss = 0.0
        for epoch in tqdm(range(self.opt.n_epochs)):  # loop over the dataset multiple times

            for batch_idx, (inputs, targets) in enumerate(self.train_loader):

                self.step = self.step + 1

                inputs, targets = inputs.cuda(), targets.long().cuda()

                # action: mixup variable lambda
                action, episode_logits, episode_log_probs = self.select_action(state)
                print('episode_logits', episode_logits, 'episode_log_probs', episode_log_probs)

                index = torch.randperm(inputs.shape[0]).cuda()
                targets_a, targets_b = targets, targets[index]
                mixed_x = action * inputs + (1 - action) * inputs[index, :]

                outputs = net(mixed_x)

                loss = action * criterion(outputs, targets_a) + (1 - action) * criterion(outputs, targets_b)
                train_loss += loss.item()
                # mixed_x, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=1.0)

                # outputs = mixup_baseline(mixed_x)
                # loss = mixup_criterion(self.loss_fn, outputs, targets_a, targets_b, lam)
                # loss = self.loss_fn(outputs, mixed_y.long())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if batch_idx % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, batch_idx + 1, running_loss / 2000))
                    running_loss = 0.0

                if epoch == 0 and batch_idx == 0:
                    # feat_sim = self.query_model()
                    # self.init_feat_sim = feat_sim
                    # feat_sim = torch.ones(self.opt.n_query_classes).cuda()
                    # print(self.init_feat_sim.mean())

                    _ = self.test(net, criterion)

                    self.init_train_loss = train_loss / self.step
                    avg_train_loss = self.init_train_loss
                    self.init_test_loss = self.best_test_loss
                    state = self.model_features(avg_train_loss)

                else:
                    avg_train_loss = train_loss / self.step
                    state = self.model_features(avg_train_loss)

            _ = self.test(net, criterion)

        print('Finished Training')

        # load best performance epoch in this training session
        # model.load_weights('weights/temp_network.h5')

        acc = self.test(net, criterion)

        # compute the reward
        reward = acc

        episode_weighted_log_probs = episode_log_probs * reward
        sum_weighted_log_probs = torch.sum(episode_weighted_log_probs).unsqueeze(dim=0)

        return sum_weighted_log_probs, episode_logits, reward

    def test(self, net, criterion):
        # evaluate the model
        correct = 0
        total = 0
        test_loss = 0
        with torch.no_grad():
            for (inputs, targets) in self.test_loader:
                inputs, targets = inputs.cuda(), targets.long().cuda()

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += inputs.size(0)
                correct += (predicted == targets).sum().item()

                test_loss += criterion(outputs, targets.long()).item()

        test_loss /= len(self.test_loader)
        if self.best_test_loss > test_loss:
            self.best_test_loss = test_loss

        acc = 100 * correct / total
        print('Accuracy of the network on the 10000 test images: {}'.format(acc))
        return acc

    def calculate_loss(self, epoch_logits: torch.Tensor, weighted_log_probs: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
            Calculates the policy "loss" and the entropy bonus
            Args:
                epoch_logits: logits of the policy network we have collected over the epoch
                weighted_log_probs: loP * W of the actions taken
            Returns:
                policy loss + the entropy bonus
                entropy: needed for logging
        """
        policy_loss = -1 * torch.mean(weighted_log_probs)

        # add the entropy bonus
        p = softmax(epoch_logits, dim=1)
        log_p = log_softmax(epoch_logits, dim=1)
        entropy = -1 * torch.mean(torch.sum(p * log_p, dim=1), dim=0)
        entropy_bonus = -1 * self.BETA * entropy

        return policy_loss + entropy_bonus, entropy

    def log(self, mode, name, value, step):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        writer.add_scalar("{}/{}/{}".format(self.experiment, mode, name), value, step)
