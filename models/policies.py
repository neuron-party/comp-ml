# https://github.com/lerrytang/train-procgen-pfrl/blob/main/policies.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.utils import *


class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=3,
                               padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=3,
                               padding=1)

    def forward(self, x):
        inputs = x
        x = torch.relu(x)
        x = self.conv0(x)
        x = torch.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):

    def __init__(self, input_shape, out_channels):
        super(ConvSequence, self).__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0],
                              out_channels=self._out_channels,
                              kernel_size=3,
                              padding=1)
        self.max_pool2d = nn.MaxPool2d(kernel_size=3,
                                       stride=2,
                                       padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool2d(x)
        x = self.res_block0(x)
        x = self.res_block1(x)
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return self._out_channels, (h + 1) // 2, (w + 1) // 2


class ImpalaCNN(nn.Module): # procgen and atari
    """Network from IMPALA paper, to work with pfrl."""

    def __init__(self, obs_space, num_outputs):

        super(ImpalaCNN, self).__init__()

        h, w, c = obs_space.shape
        shape = (c, h, w)

        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
            
        self.conv_seqs = nn.ModuleList(conv_seqs)
        self.hidden_fc = nn.Linear(in_features=shape[0] * shape[1] * shape[2],
                                   out_features=256)
        self.logits_fc = nn.Linear(in_features=256, out_features=num_outputs)
        self.value_fc = nn.Linear(in_features=256, out_features=1)
        
        # Initialize weights of logits_fc
        nn.init.orthogonal_(self.logits_fc.weight, gain=0.01)
        nn.init.zeros_(self.logits_fc.bias)

    def forward(self, obs):
        assert obs.ndim == 4
        x = obs / 255.0  # scale to 0-1
        x = x.permute(0, 3, 1, 2)  # NHWC => NCHW
        for conv_seq in self.conv_seqs:
            x = conv_seq(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(x)
        x = self.hidden_fc(x)
        x = torch.relu(x)
        logits = self.logits_fc(x)
        dist = torch.distributions.Categorical(logits=logits)
        value = self.value_fc(x)
        return dist, value

    def save_to_file(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load_from_file(self, model_path):
        self.load_state_dict(torch.load(model_path))
        
        
        
class ContinuousActionMLP(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(in_features=observation_space, out_features=256)),
            nn.Tanh(),
            layer_init(nn.Linear(in_features=256, out_features=256)),
            nn.Tanh(),
            layer_init(nn.Linear(in_features=256, out_features=1), std=1.0)
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(in_features=observation_space, out_features=256)),
            nn.Tanh(),
            layer_init(nn.Linear(in_features=256, out_features=256)),
            nn.Tanh(),
            layer_init(nn.Linear(in_features=256, out_features=action_space), std=0.01)
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_space))
        
    def forward(self, x):
        action_mean = self.actor_mean(x) # [b, action_space]
        action_logstd = self.actor_logstd.expand_as(action_mean) # [b, action_space]
        action_std = torch.exp(action_logstd)
        pi = torch.distributions.Normal(action_mean, action_std)
        value = self.critic(x) # [b, 1]
        return pi, value
        

        
class MLP(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(in_features=observation_space, out_features=64),
            nn.Tanh(),
            nn.Linear(in_features=64, out_features=64),
            nn.Tanh(),
            nn.Linear(in_features=64, out_features=action_space)
        )
        self.critic = nn.Sequential(
            nn.Linear(in_features=observation_space, out_features=64),
            nn.Tanh(),
            nn.Linear(in_features=64, out_features=64),
            nn.Tanh(),
            nn.Linear(in_features=64, out_features=1)
        )
    
    def forward(self, x):
        pi, value = self.actor(x), self.critic(x)
        pi = torch.distributions.Categorical(logits=pi)
        return pi, value
    

    
class MLP2(nn.Module): # for classic control
    def __init__(self, observation_space, action_space):
        super().__init__()
        
        self.linear1 = nn.Linear(in_features=observation_space, out_features=256)
        self.linear2 = nn.Linear(in_features=256, out_features=256)
        self.pi = nn.Linear(in_features=256, out_features=action_space)
        self.value = nn.Linear(in_features=256, out_features=1)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.tanh(self.linear1(x))
        x = self.tanh(self.linear2(x))
        pi, value = self.pi(x), self.value(x)
        pi = torch.distributions.Categorical(logits=pi)
        return pi, value