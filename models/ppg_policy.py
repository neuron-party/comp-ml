import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.utils import *


# model
class PPG_Impala(nn.Module):
    def __init__(self, envs):
        super().__init__()
        h, w, c = envs.single_observation_space.shape
        shape = (c, h, w)
        conv_seqs = []
        chans = [16, 32, 32]
        scale = 1 / np.sqrt(len(chans))  # Not fully sure about the logic behind this but its used in PPG code
        for out_channels in chans:
            conv_seq = ConvSequence(shape, out_channels, scale=scale)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)

        encodertop = nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256)
        encodertop = layer_init_normed(encodertop, norm_dim=1, scale=1.4)
        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            encodertop,
            nn.ReLU(),
        ]
        self.network = nn.Sequential(*conv_seqs)
        self.actor = layer_init_normed(nn.Linear(256, envs.single_action_space.n), norm_dim=1, scale=0.1)
        self.critic = layer_init_normed(nn.Linear(256, 1), norm_dim=1, scale=0.1)
        self.aux_critic = layer_init_normed(nn.Linear(256, 1), norm_dim=1, scale=0.1)
        
    def forward(self, x):
        assert x.ndim == 4
        x = x / 255.0
        x = x.permute(0, 3, 1, 2)
        hidden = self.network(x)
        pi_logits, value_estimate, aux_value_estimate = self.actor(hidden), self.critic(hidden), self.aux_critic(hidden)
        
        pi = torch.distributions.Categorical(logits=pi_logits)
        return pi, pi_logits, value_estimate, aux_value_estimate