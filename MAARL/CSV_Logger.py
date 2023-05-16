# if i still need to write a nested list in row: https://stackoverflow.com/questions/32855996/python-save-arbitrarily-nested-list-to-csv

import csv
import sys

import numpy as np

logs_path = 'logs/'

class CSV_Logger:
    def __init__(self, N_epochs, N_games, memory_size, batch_size, gamma, lr, anneal_lr, epsilon, entropy_coef, value_loss_coef, relevant_segments, done_after_collision, model_arch, test_mode, path):
        self.N_epochs = N_epochs
        self.N_games = N_games
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.anneal_lr = anneal_lr
        self.epsilon = epsilon
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.relevant_segments = relevant_segments
        self.done_after_collision = done_after_collision
        self.model_arch = model_arch
        self.test_mode = test_mode
        self.path = path
        if self.test_mode:
            path_extension = '_test_log.csv'
        else:
            path_extension = '_train_log.csv'
        self.path = self.path + path_extension
        self.init_logger()

    def init_logger(self):
        with open(logs_path+self.path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow(['N_Epochs'] + ['N_games'] + ['memory_size'] + ['batch_size'] + ['gamma'] + ['lr'] + ['anneal_lr'] + ['epsilon'] + ['entropy_coef'] + ['value_loss_coef'] + ['relevant_segments'] + ['done_after_collision'] + ['model_arch'])
            writer.writerow([self.N_epochs, self.N_games, self.memory_size, self.batch_size, self.gamma, self.lr, self.anneal_lr, self.epsilon, self.entropy_coef, self.value_loss_coef, self.relevant_segments, self.done_after_collision, self.model_arch])
            writer.writerow('')
            writer.writerow((['step'] + ['episode'] + ['value'] + ['action[-10°]'] + ['action[-5°]'] + ['action[0°]'] + ['action[5°]'] + ['action[10°]'] + ['reward'] + ['collision_status'] + ['length_ratio']))

    def add_rows(self, episodes, probs, values, rewards, collision_status, length_ratio):
        with open(logs_path+self.path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            for episode in range(0, len(episodes)):
                for step in range(0, len(episodes[episode])):
                    if (self.path == 'adv_1_train_log.csv') or (self.path == 'adv_1_test_log.csv'): # todo: hardcoded for now! has to be changed, so other model names in config are accepted!
                        writer.writerow([step+1, episodes[episode][step], values[episode][step], probs[episode][step][0], probs[episode][step][1], probs[episode][step][2], probs[episode][step][3], probs[episode][step][4], rewards[episode][step], collision_status[episode][step], 0])
                    else:   # for prot we have episodic RL -> one step only per episode, also we use the length_ratio thats why we differentiate here...
                        writer.writerow([step+1, episodes[episode][step], values[episode][step], probs[episode][step][0], probs[episode][step][1], probs[episode][step][2], probs[episode][step][3], probs[episode][step][4], rewards[episode][step], collision_status[episode], length_ratio[episode]])


class CSV_Reader:
    def __init__(self, path, test_mode=True):
        self.path = path
        self.test_mode = test_mode
        if test_mode:
            path_extension = '_test_log.csv'
        else:
            path_extension = '_train_log.csv'
        self.path = self.path + path_extension
        self.data_labels = []
        self.steps = []
        self.actions_0 = []
        self.actions_1 = []
        self.actions_2 = []
        self.actions_3 = []
        self.actions_4 = []
        self.values = []
        self.rewards = []
        self.collisions = []
        self.length_ratios = []
        self.read_data()

    def read_data(self):
        self.data_labels = ['step', 'episode', 'value', 'action[-10°]', 'action[-5°]', 'action[0°]', 'action[5°]', 'action[10°]', 'reward', 'collision', 'length_ratio']

        with open((logs_path+self.path), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            i = 0
            step_index = self.data_labels.index('step')
            reward_index = self.data_labels.index('reward')
            action_0_index = self.data_labels.index('action[-10°]')
            value_index = self.data_labels.index('value')
            collision_index = self.data_labels.index('collision')
            length_ratio_index = self.data_labels.index('length_ratio')

            for row in reader:
                i += 1
                if i < 5:
                    continue
                # if i % 2:   # only take every second data point
                #     continue
                self.steps.append(int(row[step_index]))
                self.rewards.append(float(row[reward_index]))
                self.actions_0.append(np.float(row[action_0_index]))
                self.actions_1.append(np.float(row[action_0_index+1]))
                self.actions_2.append(np.float(row[action_0_index+2]))
                self.actions_3.append(np.float(row[action_0_index+3]))
                self.actions_4.append(np.float(row[action_0_index+4]))
                self.values.append(np.float(row[value_index]))
                self.collisions.append(np.float(row[collision_index]))
                self.length_ratios.append(np.float(row[length_ratio_index]))

            avg = np.mean(self.rewards)
            var = np.var(self.rewards)
            print('read data.')
            print('mean:', np.mean(self.rewards))
            print('variance:', np.var(self.rewards))

            if self.test_mode:
                with open((logs_path+self.path), 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=';')
                    writer.writerow('')
                    writer.writerow((['avg'] + ['var']))
                    writer.writerow([avg, var])
