import os, random
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from buffer import ReplayBuffer
from network import Net


class Agent:
    def __init__(self, lr=0.003, input_dims=[4], env=None, gamma=0.99, n_actions=2, epsilon_greedy_start=0.5,
                 epsilon_greedy_decay=0.0002, max_size=1000000, layer1_size=64, layer2_size=64, batch_size=128, writer=None):
        self.env = env
        self.gamma = gamma
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.epsilon_greedy_start = epsilon_greedy_start
        self.epsilon_greedy_decay = epsilon_greedy_decay

        self.net = Net(lr, input_dims, n_actions=n_actions, fc1_dims=layer1_size, fc2_dims=layer2_size, name='dqn')
        self.target_net = deepcopy(self.net)
        self.target_net.load_state_dict(self.net.state_dict())
        self.target_net.eval()
        self.criterion = F.smooth_l1_loss
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.writer = writer

    def choose_action(self, state, timestep):
        epsilon = self.epsilon_greedy_start - self.epsilon_greedy_decay * timestep

        if random.random() <= epsilon:
            return self.env.action_space.sample()

        state = torch.from_numpy(state).to(self.device, torch.float)
        action = self.net.forward(state).max(0)[1].item()
        return action

    def target_update(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def model_update(self, timestep):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, states_, terminals = self.memory.sample_buffer(self.batch_size)
        state_actinon_values = self.net.forward(states.to(torch.float))
        state_actinon_values = state_actinon_values.gather(1, actions[:, 0].unsqueeze(1).to(torch.long)).squeeze(1)

        with torch.no_grad():
            next_state_values = self.target_net(states_.to(torch.float)).max(1)[0].detach()
            expected_action_values = self.gamma * next_state_values + rewards
            expected_action_values = expected_action_values * (1 - terminals.to(torch.uint8))

        loss = self.criterion(state_actinon_values, expected_action_values.to(torch.float))
        self.writer.add_scalar("loss", loss.item(), timestep)

        self.net.optimizer.zero_grad()
        loss.backward()
        for param in self.net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.net.optimizer.step()

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transtions(state, action, reward, state_, done)
