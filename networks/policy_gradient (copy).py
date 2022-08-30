import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn.functional import one_hot, log_softmax, softmax, normalize
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from .controller import Agent
from collections import deque
# from model import NASModel
import networks

from tqdm import tqdm


class PolicyGradient:
    def __init__(self, opt, train_set, val_set, test_set, writers):
        NUM_EPOCHS = 50
        ALPHA = 5e-3        # learning rate
        BATCH_SIZE = 3     # how many episodes we want to pack into an epoch
        HIDDEN_SIZE = 64    # number of hidden nodes we have in our dnn
        BETA = 0.1          # the entropy bonus multiplier
        INPUT_SIZE = 11
        ACTION_SPACE = 11
        NUM_STEPS = 4
        GAMMA = 0.99

        self.experiment = "policy gradient"

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

        self.train = train_set
        self.val = val_set
        self.test = test_set

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
        self.agent = Agent(self.opt, self.INPUT_SIZE, self.HIDDEN_SIZE, self.NUM_STEPS, device=self.DEVICE).cuda()

        self.adam = optim.Adam(params=self.agent.parameters(), lr=self.ALPHA)
        self.total_rewards = deque([], maxlen=100)

        print(self.agent)

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
        # Init state
        init_state = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

        # get the action logits from the agent - (preferences)
        episode_logits = self.agent(torch.tensor(init_state).float().to(self.DEVICE))

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
        # generate a submodel given predicted actions
        # net = NASModel(action)
        net = networks.CNN(in_channels=self.opt.channels, num_classes=self.opt.n_classes).cuda()
        #net = Net()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        running_loss = 0.0
        for epoch in tqdm(range(self.opt.n_epochs)):  # loop over the dataset multiple times

            for batch_idx, (inputs, targets) in enumerate(self.train):
                inputs, targets = inputs.cuda(), targets.long().cuda()

                lam = action[epoch]

                index = torch.randperm(inputs.shape[0]).cuda()
                targets_a, targets_b = targets, targets[index]
                mixed_x = lam * inputs + (1 - lam) * inputs[index, :]

                outputs = net(mixed_x)

                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)

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

        print('Finished Training')

        # load best performance epoch in this training session
        # model.load_weights('weights/temp_network.h5')

        # evaluate the model
        correct = 0
        total = 0
        with torch.no_grad():
            for (inputs, targets) in self.test:
                inputs, targets = inputs.cuda(), targets.long().cuda()

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += inputs.size(0)
                correct += (predicted == targets).sum().item()

        acc = 100 * correct / total
        print('Accuracy of the network on the 10000 test images: {}'.format(acc))

        # compute the reward
        reward = acc

        episode_weighted_log_probs = episode_log_probs * reward
        sum_weighted_log_probs = torch.sum(episode_weighted_log_probs).unsqueeze(dim=0)

        return sum_weighted_log_probs, episode_logits, reward

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
