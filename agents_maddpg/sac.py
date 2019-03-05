import torch
import torch.optim as optim
from agents_maddpg.model import TanhGaussianActorCritic
from agents_maddpg.utils import soft_update
import numpy as np


class SAC():
    """Interacts with and learns from the environment."""
    
    def __init__(self, network: TanhGaussianActorCritic, memory,device,
                LR_CRITIC,
                LR_ACTOR,
                LR_ALPHA,
                UPDATE_EVERY,
                UPDATE_LOOP,
                TAU,
                TRANSFER_EVERY,
                WEIGHT_DECAY,
                TARGET_ENTROPY,
                 GAMMA):
        
        # Actor networks
        self.network = network

        self.set_optimizer_hyperparameters(LR_CRITIC, LR_ACTOR, LR_ALPHA, WEIGHT_DECAY)
        
        self.device = device
        
        # Replay memory
        self.memory = memory
        # Initialize time steps (for updating every UPDATE_EVERY steps)
        self.u_step = 0

        self.TAU = TAU
        self.UPDATE_EVERY = UPDATE_EVERY
        self.TRANSFER_EVERY = TRANSFER_EVERY
        self.UPDATE_LOOP = UPDATE_LOOP
        self.TARGET_ENTROPY = TARGET_ENTROPY
        self.GAMMA = GAMMA

        self.policy_loss = 0
        self.estimation = 0
        self.critics_losses = np.zeros(len(self.network.critics_local))
        self.loss_function = torch.nn.MSELoss()   # SmoothL1Loss(reduce=None)

    def set_optimizer_hyperparameters(self, LR_CRITIC, LR_ACTOR, LR_ALPHA, WEIGHT_DECAY):
        self.actor_optim = optim.Adam(self.network.actor.parameters(), lr=LR_ACTOR, weight_decay=WEIGHT_DECAY)
        self.critic_optims = [optim.Adam(critic.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY) for critic in self.network.critics_local]
        self.alpha_optim = optim.Adam([self.network.log_alpha], lr=LR_ALPHA, weight_decay=WEIGHT_DECAY)
        self.WEIGHT_DECAY = WEIGHT_DECAY
        self.LR_CRITIC = LR_CRITIC
        self.LR_ACTOR = LR_ACTOR
        self.LR_ALPHA = LR_ALPHA

    
    def act(self, states):
        """Returns actions of each actor for given states.
        
        Params
        ======
            state (array_like): current states
        """
        ret = None

        with torch.no_grad():
            self.network.eval()
            states = torch.from_numpy(states).float().unsqueeze(0).to(self.device)
            ret = self.network(states).squeeze().cpu().data.numpy()
            self.network.train()
        return ret

    def test(self, states):
        ret = None

        with torch.no_grad():
            self.network.eval()
            states = torch.from_numpy(states).float().unsqueeze(0).to(self.device)
            ret = self.network.actor.test(states).squeeze().cpu().data.numpy()
            self.network.train()
        return ret
    
    def step(self, states, actions, rewards, next_states, dones):
        # Save experience in replay memory
        # self.memory.add(states, actions, rewards, next_states, dones)
        self.add_to_memory(states, actions, rewards, next_states, dones)

        # Learn every UPDATE_EVERY timesteps
        self.u_step = (self.u_step + 1) % self.UPDATE_EVERY

        if len(self.memory) > self.memory.batch_size and self.u_step == 0:
            t_step=0
            for _ in range(self.UPDATE_LOOP):
                self.learn()
                t_step=(t_step + 1) % self.TRANSFER_EVERY
                if t_step == 0:
                    self.update_target()

    def add_to_memory(self, states, actions, rewards, next_states, dones):
        for i in range(len(states)):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

    def update_target(self, tau=None):
        tau = tau or self.TAU
        for local, target in zip(self.network.critics_local, self.network.critics_target):
            soft_update(local, target, tau)

    def learn(self):
        states, actions, rewards, next_states, dones = self.memory.sample()
        alpha = self.network.log_alpha.exp().detach()

        # optimize the critic
        next_actions, next_log_probs = self.network.get_with_probabilities(next_states)
        concatenated = torch.cat((next_states, next_actions), dim=-1)
        next_q_targets = torch.stack([critic(concatenated) for critic in self.network.critics_target])
        next_min_q, _ = next_q_targets.min(dim=0)
        assert next_min_q.shape == next_log_probs.shape, " next_min_q {} != next_log_probs {}".format(next_min_q.shape, next_log_probs.shape)
        dones = dones.unsqueeze(-1)
        assert next_min_q.shape == dones.shape, " next_min_q {} != dones {}".format(next_min_q.shape, dones.shape)
        rewards = rewards.unsqueeze(-1)
        assert next_min_q.shape == rewards.shape, " next_min_q {} != rewards {}".format(next_min_q.shape, rewards.shape)
        next_value = next_min_q - alpha * next_log_probs
        target = (rewards + self.GAMMA * (1 - dones) * next_value).detach()

        for critic, optim, i in zip(self.network.critics_local, self.critic_optims,range(len(self.network.critics_local))):
            concatenated = torch.cat((states, actions), dim=-1)
            current = critic(concatenated)
            optim.zero_grad()
            loss = self.loss_function(current, target)
            loss.backward()
            self.critics_losses[i]=loss.detach().cpu().numpy().item()
            optim.step()

        new_actions, new_log_probs = self.network.get_with_probabilities(states)
        # optimize alpha
        alpha_loss = - self.network.log_alpha * (new_log_probs + self.TARGET_ENTROPY).detach()
        self.alpha_optim.zero_grad()
        alpha_loss.mean().backward()
        self.alpha_optim.step()

        # optimize the actor
        concatenated = torch.cat((states, new_actions), dim=-1)
        q_locals = torch.stack([critic(concatenated) for critic in self.network.critics_local])
        min_q_locals, _ = q_locals.min(dim=0)
        assert min_q_locals.shape == new_log_probs.shape, " min_q_locals {} != new_log_probs {}".format(
            min_q_locals.shape, new_log_probs.shape)
        self.estimation = min_q_locals.mean().detach().cpu().numpy().item()
        policy_kl_losses = alpha * new_log_probs - min_q_locals
        self.actor_optim.zero_grad()
        loss = policy_kl_losses.mean()
        loss.backward()
        self.policy_loss = loss.detach().cpu().numpy().item()
        self.actor_optim.step()
