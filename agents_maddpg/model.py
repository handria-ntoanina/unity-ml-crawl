import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

class TanhGaussianActorCritic(nn.Module):
    def __init__(self, state_size, action_size, q_number, activation=F.relu):
        super().__init__()
        # This need a change in I:\MyDev\Anaconda3\envs\drlnd\Lib\site-packages\torch\distributions\utils.py
        # at the line 70, check the type using isinstance instead of __class__.__name__
        self.activation = activation
        self.state_size = state_size
        self.action_size = action_size

        self.log_alpha = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.actor = TanhGaussian(state_size, action_size, activation=activation)
        self.critics_local = [FullyConnected([state_size + action_size, 128, 128, 1], activation=activation) for i in
                              range(q_number)]
        for fc in self.critics_local:
            layer_init(fc.hidden_layers[-1], 1e-3)
        self.critics_target = [FullyConnected([state_size + action_size, 128, 128, 1], activation=activation) for i in
                               range(q_number)]
        for fc in self.critics_target:
            layer_init(fc.hidden_layers[-1], 1e-3)

    def forward(self, state):
        return self.actor(state)

    def get_with_probabilities(self, state):
        return self.actor.get_with_probabilities(state)

class TanhGaussianActorCriticValue(nn.Module):
    def __init__(self, state_size, action_size, q_number, activation=F.relu):
        super().__init__()
        # This need a change in I:\MyDev\Anaconda3\envs\drlnd\Lib\site-packages\torch\distributions\utils.py
        # at the line 70, check the type using isinstance instead of __class__.__name__
        self.activation = activation
        self.state_size = state_size
        self.action_size = action_size

        self.log_alpha = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.actor = TanhGaussian(state_size, action_size, activation=activation)
        self.value_local = FullyConnected([state_size, 128, 128, 1], activation=activation)
        self.value_target = FullyConnected([state_size, 128, 128, 1], activation=activation)
        self.critics_local = [FullyConnected([state_size + action_size, 128, 128, 1], activation=activation) for i in
                              range(q_number)]

    def forward(self, state):
        return self.actor(state)

    def get_with_probabilities(self, state):
        return self.actor.get_with_probabilities(state)

class TanhGaussian(nn.Module):
    def __init__(self, state_size, action_size, activation=F.relu):
        super().__init__()
        # This need a change in I:\MyDev\Anaconda3\envs\drlnd\Lib\site-packages\torch\distributions\utils.py
        # at the line 70, check the type using isinstance instead of __class__.__name__
        self.activation = activation
        self.fc = FullyConnected([state_size, 128, 128], activation=activation)
        self.mean_linear = nn.Linear(128, action_size)
        layer_init(self.mean_linear)
        self.log_std_linear = nn.Linear(128, action_size)
        layer_init(self.log_std_linear)
        self.log_std_min = -20
        self.log_std_max = 2
        self.action_size = action_size

    def forward(self, state):
        action, _ = self.get_with_probabilities(state)
        return action

    def test(self, state):
        x = self.activation(self.fc(state))
        return torch.tanh(self.mean_linear(x))

    def get_with_probabilities(self, state):
        x = self.activation(self.fc(state))
        mean = self.mean_linear(x)
        log_std = torch.clamp(self.log_std_linear(x), self.log_std_min, self.log_std_max)
        std = log_std.exp()
        actions = torch.distributions.Normal(0, torch.ones(self.action_size)).sample()*std + mean
        actions_tanh = torch.tanh(actions)
        # This is an approximator of the log likelihood of tanh(actions)
        log_prob = torch.distributions.Normal(mean, std).log_prob(actions) - torch.log( 1 - actions_tanh.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return actions_tanh, log_prob

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
