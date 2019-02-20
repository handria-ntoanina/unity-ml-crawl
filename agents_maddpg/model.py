import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from agents_maddpg.utils import soft_update

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, activation=F.relu):
        super().__init__()
        self.actor = FullyConnected([state_size, 112, 112,  action_size], activation=activation)
        self.critic = FullyConnected([state_size + action_size, 112, 112, 1], activation=activation)
    
    def forward(self, state):
        return F.tanh(self.actor(state))
    
    def estimate(self, state, action):
        """
            The expected input here is the boosted information. That are the states seen by all agents and their actions
        """
        concatenated = torch.cat((state, action), dim=-1)
        return self.critic(concatenated)

class FullyConnected(nn.Module):
    def __init__(self, hidden_layers_size, activation=F.relu):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of action space
        """
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.extend([layer_init(nn.Linear(hidden_layers_size[i], hidden_layers_size[i + 1])) for i in range(len(hidden_layers_size) - 1)])
        layer_init(self.hidden_layers[-1], 1e-3)
        self.activation = activation
        
    def forward(self, x):
        """Build a network that maps state to actions."""
        for i in range(len(self.hidden_layers) - 1):
            linear = self.hidden_layers[i]
            x = self.activation(linear(x))
        return self.hidden_layers[-1](x)
