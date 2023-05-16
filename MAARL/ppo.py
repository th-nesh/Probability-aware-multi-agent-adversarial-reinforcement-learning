# references used for implementing PPO-RL: (mostly a mix of those two references and some adjusting...)
# https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/PPO/torch
# https://github.com/vwxyzjn/PPO-Implementation-Deep-Dive

import numpy as np
import torch
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    this function initializes the passed layer with an "orthogonal" initialization
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOMemory:
    """
    this class represents the memory of the agent
    all the attributes (except batch_size) are lists which are filled with the relevant data for every step.
    attributes:
    states - list of all "remembered" states
    probs - list of all "remembered" probs -> probs are the output values (softmax) of the actor (of the agent)
    vals - list of all "remembered" values -> values as they were estimated by the critic (output of critic) (of the agent)
    actions - list of all "remembered" actions -> actions that were taken at the corresponding step
    rewards - list of all "remembered" reward values
    entropy - list of all "remembered" entropy values
    dones - list of all "remembered" done values -> tells us if it was the last step in a particular episode
    """
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.entropy = []
        self.dones = []

        self.batch_size = batch_size

    def get_episodes(self):
        return self.states, \
               self.actions, \
               self.probs, \
               self.vals, \
               self.rewards, \
               self.entropy, \
               self.dones

    # if the amount of steps in memory is not aliquot by batch_size, the last batch will simply be smaller - that should be okay.
    def generate_batches(self):
        """
        generates the batches -> lists of random permutation of indexes, which will be used in training
        """
        all_states = np.concatenate(self.states)

        n_states = len(all_states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        # this we have to use if we use batch normalization
        if len(batches[-1]) == 1:
            batches.pop(-1)

        return batches

    def store_memory(self, states, actions, probs, vals, rewards, entropy, dones):
        self.states.append(states)
        self.actions.append(actions)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(rewards)
        self.entropy.append(entropy)
        self.dones.append(dones)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.entropy = []
        self.dones = []
        self.vals = []


class ActorNetwork(nn.Module):
    """
    actor (neural network) component of the actor-critic...
    returns the probabilities of which action is preferred to take by the neural network
    model_arch is just for faster testing of different hyperparameters. usually model_arch will just be "normal"
        --!! important note !!--: this is for now only implemented for input size (map size) of 200x200 pixels
                              if a different size is used the tesor size after the convolutional layer has to be
                              calculated and the input size for the linear layer after flatten(..) has to be adjusted.
    """
    def __init__(self, n_actions, lr, model_arch='normal'):
        super(ActorNetwork, self).__init__()

        self.model_arch = model_arch

        if model_arch == 'small':
            print('small')
            self.actor = nn.Sequential(
                layer_init(nn.Conv2d(1, 8, (3, 3))),
                nn.LeakyReLU(),
                #nn.BatchNorm2d(8),
                layer_init(nn.Conv2d(8, 8, (3, 3))),
                nn.LeakyReLU(),
                #nn.BatchNorm2d(8),
                layer_init(nn.Conv2d(8, 16, (5, 5), stride=(2, 2))),
                nn.LeakyReLU(),
                #nn.BatchNorm2d(16),

                layer_init(nn.Conv2d(16, 16, (3, 3))),
                nn.LeakyReLU(),
                #nn.BatchNorm2d(16),
                layer_init(nn.Conv2d(16, 16, (3, 3))),
                nn.LeakyReLU(),
                #nn.BatchNorm2d(16),
                layer_init(nn.Conv2d(16, 16, (5, 5), stride=(2, 2))),
                nn.LeakyReLU(),
                #nn.BatchNorm2d(16),

                layer_init(nn.Conv2d(16, 16, (3, 3))),
                nn.LeakyReLU(),
                #nn.BatchNorm2d(16),
                layer_init(nn.Conv2d(16, 16, (3, 3))),
                nn.LeakyReLU(),
                #nn.BatchNorm2d(16),
                layer_init(nn.Conv2d(16, 16, (3, 3), stride=(2, 2))),
                nn.LeakyReLU(),
                #nn.BatchNorm2d(16),

                nn.Flatten(),

                layer_init(nn.Linear(16 * 19 * 19, 128)),
                nn.LeakyReLU(),
                #nn.BatchNorm1d(128),
                layer_init(nn.Linear(128, n_actions), 0.01),
                nn.Softmax(dim=1)
            )
        elif model_arch == 'normal':
            print('normal')
            self.actor = nn.Sequential(
                layer_init(nn.Conv2d(1, 8, (3, 3))),
                nn.LeakyReLU(),
                nn.BatchNorm2d(8),
                layer_init(nn.Conv2d(8, 8, (3, 3))),
                nn.LeakyReLU(),
                nn.BatchNorm2d(8),
                layer_init(nn.Conv2d(8, 16, (5, 5), stride=(2, 2))),
                nn.LeakyReLU(),
                nn.BatchNorm2d(16),

                layer_init(nn.Conv2d(16, 16, (3, 3))),
                nn.LeakyReLU(),
                nn.BatchNorm2d(16),
                layer_init(nn.Conv2d(16, 16, (3, 3))),
                nn.LeakyReLU(),
                nn.BatchNorm2d(16),
                layer_init(nn.Conv2d(16, 16, (5, 5), stride=(2, 2))),
                nn.LeakyReLU(),
                nn.BatchNorm2d(16),

                layer_init(nn.Conv2d(16, 16, (3, 3))),
                nn.LeakyReLU(),
                nn.BatchNorm2d(16),
                layer_init(nn.Conv2d(16, 16, (3, 3))),
                nn.LeakyReLU(),
                nn.BatchNorm2d(16),
                layer_init(nn.Conv2d(16, 16, (3, 3), stride=(2, 2))),
                nn.LeakyReLU(),
                nn.BatchNorm2d(16),

                nn.Flatten(),

                layer_init(nn.Linear(16 * 19 * 19, 128)),   # 200x200 map
                # layer_init(nn.Linear(16 * 14 * 14, 128)),   # 160x160 map
                nn.LeakyReLU(),
                nn.BatchNorm1d(128),
                layer_init(nn.Linear(128, n_actions), 0.01),
                nn.Softmax(dim=1)
            )
        elif model_arch == 'grande':
            print('grande')
            self.actor = nn.Sequential(
                layer_init(nn.Conv2d(1, 16, (3, 3))),
                nn.LeakyReLU(),
                nn.BatchNorm2d(16),
                layer_init(nn.Conv2d(16, 16, (3, 3))),
                nn.LeakyReLU(),
                nn.BatchNorm2d(16),
                layer_init(nn.Conv2d(16, 32, (5, 5), stride=(2, 2))),
                nn.LeakyReLU(),
                nn.BatchNorm2d(32),

                layer_init(nn.Conv2d(32, 32, (3, 3))),
                nn.LeakyReLU(),
                nn.BatchNorm2d(32),
                layer_init(nn.Conv2d(32, 32, (3, 3))),
                nn.LeakyReLU(),
                nn.BatchNorm2d(32),
                layer_init(nn.Conv2d(32, 64, (5, 5), stride=(2, 2))),
                nn.LeakyReLU(),
                nn.BatchNorm2d(64),

                layer_init(nn.Conv2d(64, 64, (3, 3))),
                nn.LeakyReLU(),
                nn.BatchNorm2d(64),
                layer_init(nn.Conv2d(64, 64, (3, 3))),
                nn.LeakyReLU(),
                nn.BatchNorm2d(64),
                layer_init(nn.Conv2d(64, 64, (3, 3))),
                nn.LeakyReLU(),
                nn.BatchNorm2d(64),

                nn.MaxPool2d((2, 2), stride=(2, 2)),
                nn.Flatten(),

                layer_init(nn.Linear(64 * 14 * 14, 128)),
                nn.LeakyReLU(),
                nn.BatchNorm1d(128),
                layer_init(nn.Linear(128, n_actions), 0.01),
                nn.Softmax(dim=1)
            )

        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=1e-5)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        #moving to the device 
        self.to(self.device)

    def forward(self, x):
        x = self.actor(x)

        return x

    def save_checkpoint(self, path, name):
        T.save(self.state_dict(), path+'actor_'+name)
        print('saving:', name)

    def load_checkpoint(self, path, name):
        self.load_state_dict(T.load(path+'actor_'+name, map_location=T.device('cpu')))


class CriticNetwork(nn.Module):
    def __init__(self, lr, model_arch='normal'):
        """
    critic (neural network) component of the actor-critic...
    returns the value (value as in the RL-notation) which the neural network estimates for a given state
    model_arch is just for faster testing of different hyperparameters. usually model_arch will just be "normal"
    --!! important note !!--: this is for now only implemented for input size (map size) of 200x200 pixels
                              if a different size is used the tesor size after the convolutional layer has to be
                              calculated and the input size for the linear layer after flatten(..) has to be adjusted.
    """
        super(CriticNetwork, self).__init__()
        self.model_arch = model_arch

        if model_arch == 'small':
            print('small')
            self.critic = nn.Sequential(
                layer_init(nn.Conv2d(1, 8, (3, 3))),
                nn.LeakyReLU(),
                #nn.BatchNorm2d(8),
                layer_init(nn.Conv2d(8, 8, (3, 3))),
                nn.LeakyReLU(),
                #nn.BatchNorm2d(8),
                layer_init(nn.Conv2d(8, 16, (5, 5), stride=(2, 2))),
                nn.LeakyReLU(),
                #nn.BatchNorm2d(16),

                layer_init(nn.Conv2d(16, 16, (3, 3))),
                nn.LeakyReLU(),
                #nn.BatchNorm2d(16),
                layer_init(nn.Conv2d(16, 16, (3, 3))),
                nn.LeakyReLU(),
                #nn.BatchNorm2d(16),
                layer_init(nn.Conv2d(16, 16, (5, 5), stride=(2, 2))),
                nn.LeakyReLU(),
                #nn.BatchNorm2d(16),

                layer_init(nn.Conv2d(16, 16, (3, 3))),
                nn.LeakyReLU(),
                #nn.BatchNorm2d(16),
                layer_init(nn.Conv2d(16, 16, (3, 3))),
                nn.LeakyReLU(),
                #nn.BatchNorm2d(16),
                layer_init(nn.Conv2d(16, 16, (3, 3), stride=(2, 2))),
                nn.LeakyReLU(),
                #nn.BatchNorm2d(16),

                nn.Flatten(),

                layer_init(nn.Linear(16 * 19 * 19, 128)),
                nn.LeakyReLU(),
                #nn.BatchNorm1d(128),
                layer_init(nn.Linear(128, 1), 1.)
            )
        elif model_arch == 'normal':
            print('normal')
            self.critic = nn.Sequential(
                layer_init(nn.Conv2d(1, 8, (3, 3))),
                nn.LeakyReLU(),
                nn.BatchNorm2d(8),
                layer_init(nn.Conv2d(8, 8, (3, 3))),
                nn.LeakyReLU(),
                nn.BatchNorm2d(8),
                layer_init(nn.Conv2d(8, 16, (5, 5), stride=(2, 2))),
                nn.LeakyReLU(),
                nn.BatchNorm2d(16),

                layer_init(nn.Conv2d(16, 16, (3, 3))),
                nn.LeakyReLU(),
                nn.BatchNorm2d(16),
                layer_init(nn.Conv2d(16, 16, (3, 3))),
                nn.LeakyReLU(),
                nn.BatchNorm2d(16),
                layer_init(nn.Conv2d(16, 16, (5, 5), stride=(2, 2))),
                nn.LeakyReLU(),
                nn.BatchNorm2d(16),

                layer_init(nn.Conv2d(16, 16, (3, 3))),
                nn.LeakyReLU(),
                nn.BatchNorm2d(16),
                layer_init(nn.Conv2d(16, 16, (3, 3))),
                nn.LeakyReLU(),
                nn.BatchNorm2d(16),
                layer_init(nn.Conv2d(16, 16, (3, 3), stride=(2, 2))),
                nn.LeakyReLU(),
                nn.BatchNorm2d(16),

                nn.Flatten(),

                layer_init(nn.Linear(16 * 19 * 19, 128)),   # 200x200 map
                # layer_init(nn.Linear(16 * 14 * 14, 128)),   # 160x160 map
                nn.LeakyReLU(),
                nn.BatchNorm1d(128),
                layer_init(nn.Linear(128, 1), 1.)
            )
        elif model_arch == 'grande':
            print('grande')
            self.critic = nn.Sequential(
                layer_init(nn.Conv2d(1, 16, (3, 3))),
                nn.LeakyReLU(),
                nn.BatchNorm2d(16),
                layer_init(nn.Conv2d(16, 16, (3, 3))),
                nn.LeakyReLU(),
                nn.BatchNorm2d(16),
                layer_init(nn.Conv2d(16, 32, (5, 5), stride=(2, 2))),
                nn.LeakyReLU(),
                nn.BatchNorm2d(32),

                layer_init(nn.Conv2d(32, 32, (3, 3))),
                nn.LeakyReLU(),
                nn.BatchNorm2d(32),
                layer_init(nn.Conv2d(32, 32, (3, 3))),
                nn.LeakyReLU(),
                nn.BatchNorm2d(32),
                layer_init(nn.Conv2d(32, 64, (5, 5), stride=(2, 2))),
                nn.LeakyReLU(),
                nn.BatchNorm2d(64),

                layer_init(nn.Conv2d(64, 64, (3, 3))),
                nn.LeakyReLU(),
                nn.BatchNorm2d(64),
                layer_init(nn.Conv2d(64, 64, (3, 3))),
                nn.LeakyReLU(),
                nn.BatchNorm2d(64),
                layer_init(nn.Conv2d(64, 64, (3, 3))),
                nn.LeakyReLU(),
                nn.BatchNorm2d(64),

                nn.MaxPool2d((2, 2), stride=(2, 2)),
                nn.Flatten(),

                layer_init(nn.Linear(64 * 14 * 14, 128)),
                nn.LeakyReLU(),
                nn.BatchNorm1d(128),
                layer_init(nn.Linear(128, 1), 1.)
            )

        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=1e-5)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = self.critic(x)

        return x

    def save_checkpoint(self, path, name):
        T.save(self.state_dict(), path+'critic_'+name)
        print('saving:', name)

    def load_checkpoint(self, path, name):
        self.load_state_dict(T.load(path+'critic_'+name, map_location=T.device('cpu')))


class Agent:
    """
    Agent class consists of both, actor and critic network and considers a bunch of hyperparameters
    references used for implementing PPO-RL: (mostly a mix of those two references and some adjusting...)
        https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/PPO/torch
        https://github.com/vwxyzjn/PPO-Implementation-Deep-Dive
    """
    def __init__(self, n_actions, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, n_epochs=10, model_arch='normal', epsilon=0.5, entropy_coef=0.01, value_loss_coef=0.5):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef

        self.actor = ActorNetwork(n_actions, alpha, model_arch=model_arch)
        self.critic = CriticNetwork(alpha, model_arch=model_arch)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, entropy, done):
        self.memory.store_memory(state, action, probs, vals, reward, entropy, done)

    def save_models(self, path, name):
        print('... saving models ...')
        self.actor.save_checkpoint(path, name)
        self.critic.save_checkpoint(path, name)

    def load_models(self, path, name):
        print('... loading models ...')
        self.actor.load_checkpoint(path, name)
        self.critic.load_checkpoint(path, name)

    def choose_action(self, observation, test_mode=False):
        # exand obs from (hxw) to (1x1xhxw)
        state = T.unsqueeze(T.unsqueeze(T.from_numpy(observation), 0), 0).to(self.actor.device)
        self.actor.eval()
        self.critic.eval()
        raw_probs = self.actor(state)
        dist = Categorical(raw_probs)
        value = self.critic(state)

        #forcing the agent to choose the action based on the probality distribution in dist
        if test_mode:
            action_index_choose = T.squeeze(torch.argmax(raw_probs, dim=1).to(self.actor.device))
        else:
            if self.epsilon == -1:
                action_index_choose = dist.sample()
            else:
                r = np.random.rand()
                if (r < self.epsilon):
                    action_index_choose = dist.sample()
                else:
                    action_index_choose = torch.from_numpy(np.array(np.random.randint(self.n_actions))).to(self.actor.device)

        probs = T.squeeze(dist.log_prob(action_index_choose)).item()
        action_index_choose = T.squeeze(action_index_choose).item()
        value = T.squeeze(value).item()

        return action_index_choose, probs, value, raw_probs

    def learn(self, test_mode=False):
        self.actor.train()
        self.critic.train()

        if test_mode:
            self.memory.clear_memory()
            return 0

        for _ in range(self.n_epochs):
            state_arr_list, action_arr_list, old_prob_arr_list, vals_arr_list, \
            reward_arr_list, entropy_arr_list, dones_arr_list = \
                self.memory.get_episodes()

            all_states = np.concatenate(state_arr_list)
            all_actions = np.concatenate(action_arr_list)
            all_old_probs = np.concatenate(old_prob_arr_list)
            all_vals = np.concatenate(vals_arr_list)
            all_entropy = np.concatenate(entropy_arr_list)

            all_advantages = []

            for i in range(0, len(state_arr_list)):

                values = np.array(vals_arr_list[i])
                reward_arr = np.array(reward_arr_list[i])
                dones_arr = np.array(dones_arr_list[i])

                advantages = np.zeros(len(reward_arr), dtype=np.float32)
                for t in range(len(reward_arr)-1):
                    discount = 1
                    a_t = 0
                    for k in range(t, len(reward_arr)-1):
                        a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*(1-int(dones_arr[k])) - values[k])
                        discount *= self.gamma*self.gae_lambda
                    advantages[t] = a_t
                advantages[-1] = reward_arr[-1]-values[-1]  # the last value has yet to be set
                all_advantages.append(advantages)

            all_advantages = np.concatenate(all_advantages)
            all_advantages = T.from_numpy(all_advantages).to(self.actor.device)

            all_vals = T.from_numpy(all_vals).to(self.actor.device)

            batches = self.memory.generate_batches()
            for batch in batches:
                states_batch = np.expand_dims(all_states[batch], axis=1)

                states_batch = T.from_numpy(states_batch).to(self.actor.device)
                old_probs_batch = T.from_numpy(all_old_probs[batch]).to(self.actor.device)
                actions_batch = T.from_numpy(all_actions[batch]).to(self.actor.device)

                raw_probs = self.actor(states_batch)
                dist = Categorical(raw_probs)

                critic_value = self.critic(states_batch)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions_batch)

                prob_ratio = new_probs.exp() / old_probs_batch.exp()
                #prob_ratio = (new_probs - old_probs).exp()

                # advantage_normalization
                #advantages_batch = (all_advantages[batch]-all_advantages[batch].mean())/(all_advantages[batch].std() + 1e-8)
                advantages_batch = all_advantages[batch]

                # ppo policy clipping
                weighted_probs = advantages_batch * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                                                 1+self.policy_clip)*advantages_batch
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = (advantages_batch + all_vals[batch]).detach()

                # ppo value clipping
                value_clipping = True
                if value_clipping:
                    critic_loss_unclipped = (returns-critic_value)**2
                    value_clipped = all_vals[batch] + T.clamp(critic_value - all_vals[batch], -self.policy_clip, self.policy_clip)
                    value_loss_clipped = (returns - value_clipped)**2
                    value_loss_max = T.max(critic_loss_unclipped, value_loss_clipped)
                    critic_loss = 0.5*self.value_loss_coef * value_loss_max.mean()
                else:
                    critic_loss = 0.5*self.value_loss_coef * ((returns-critic_value)**2).mean()

                entropy_loss = all_entropy[batch].mean()

                actor_loss = actor_loss - (self.entropy_coef * entropy_loss)

                # total_loss = actor_loss - self.entropy_coef * entropy_loss + critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                # total_loss.backward()
                actor_loss.backward()
                critic_loss.backward()
                T.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                T.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()
