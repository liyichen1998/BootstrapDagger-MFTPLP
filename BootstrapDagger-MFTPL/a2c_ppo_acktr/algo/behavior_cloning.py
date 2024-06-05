# prerequisites
import copy
import glob
import sys
import os
import time
from collections import deque

import gym

import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class BehaviorCloning:
    def __init__(self, policy,  device, batch_size=None, lr=None, annotated_dataset=None,
         num_batches=np.float('inf'), training_data_split=None, envs=None, ensemble_size=None):
        super(BehaviorCloning, self).__init__()

        self.actor_critic = policy

        self.optimizer  = torch.optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.device = device
        self.lr = lr
        self.batch_size = batch_size

        datasets = annotated_dataset.load_demo_data(training_data_split, batch_size, ensemble_size)
        self.trdata = datasets['trdata']
        self.tedata = datasets['tedata']

        self.num_batches = num_batches
        self.action_space = envs.action_space

    def update(self, update=True, data_loader_type=None):
        if data_loader_type == 'train':
            data_loaders = self.trdata
        elif data_loader_type == 'test':
            data_loaders = self.tedata
        else:
            raise Exception("Unknown Data loader specified")

        total_loss = 0
        # for batch_idx, batch in enumerate(data_loader, 1):
        # for ensemble_idx, data_loader in enumerate(data_loader):
        #     for batch_idx, (states, actions) in enumerate(data_loader, 1):
        #         ensemble_states  = states.float()
        #         ensemble_actions = actions.float().to(self.device)
        for batch_idx, data_loaders_batch in enumerate(zip(*data_loaders), 1):
            states = torch.stack([batch[0] for batch in data_loaders_batch]).to(self.device)
            expert_actions = torch.stack([batch[1] for batch in data_loaders_batch]).to(self.device)
            # print(states.shape)  # Should be [ensemble_size, batch_size, feature_dim_obs]
            # print(expert_actions.shape)  # Should be [ensemble_size, batch_size, feature_dim_acs]

            self.optimizer.zero_grad()


            # dynamic_batch_size = states.shape[0]
            # try:
            #     # Regular Behavior Cloning
            #     pred_actions = self.actor_critic.get_action(states,deterministic=True).view(dynamic_batch_size, -1)
            # except AttributeError:
            # Ensemble Behavior Cloning
            pred_actions = self.actor_critic(states) #.view(dynamic_batch_size, -1)
            # print(pred_actions.shape)
            # raise Exception("debug point")

            if isinstance(self.action_space, gym.spaces.Box):
                # debug and test
                # pred_array = np.array([["{:.4f}".format(i) for i in row] for row in pred_actions.cpu().detach().numpy()])
                # print('intraining learner',pred_array)
                # print('expert',expert_actions.cpu())
                # print('states',states.cpu())
                # raise Exception("debug point")

                pred_actions = torch.clamp(pred_actions, self.action_space.low[0],self.action_space.high[0])
                expert_actions = torch.clamp(expert_actions.float(), self.action_space.low[0],self.action_space.high[0])
                loss = F.mse_loss(pred_actions, expert_actions)
            elif isinstance(self.action_space, gym.spaces.discrete.Discrete):
                raise NotImplementedError
                loss = F.cross_entropy(pred_actions, expert_actions.flatten().long())
            elif self.action_space.__class__.__name__ == "MultiBinary":
                raise NotImplementedError
                loss = torch.binary_cross_entropy_with_logits(pred_actions, expert_actions).mean()

            if update:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()

            if batch_idx >= self.num_batches:
                break

        return (total_loss / batch_idx)

    def reset(self):
        self.optimizer  = torch.optim.Adam(self.actor_critic.parameters(), lr=self.lr)

