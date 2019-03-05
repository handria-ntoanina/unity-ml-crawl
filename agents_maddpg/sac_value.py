import torch
import torch.optim as optim
from agents_maddpg.model import TanhGaussianActorCriticValue
from agents_maddpg.utils import soft_update
import numpy as np


class SAC_Value():
    """Interacts with and learns from the environment."""
    
    def __init__(self, network: TanhGaussianActorCriticValue, memory,device,
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
        self.critic_optims =[optim.Adam(critic.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY) for critic in self.network.critics_local]
        self.value_optim =optim.Adam(self.network.value_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
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



    def step(self, states, actions, rewards, next_states, dones):
        # Save experience in replay memory
        # self.memory.add(states, actions, rewards, next_states, dones)
        self.add_to_memory(actions, dones, next_states, rewards, states)

        # Learn every UPDATE_EVERY timesteps
        self.u_step = (self.u_step + 1) % self.UPDATE_EVERY

        if len(self.memory) > self.memory.batch_size and self.u_step == 0:
            t_step=0
            for _ in range(self.UPDATE_LOOP):
                self.learn()

    def add_to_memory(self, actions, dones, next_states, rewards, states):
        for i in range(len(states)):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

    def update_target(self, tau=None):
        tau = tau or self.TAU
        soft_update(self.network.value_local, self.network.value_target, tau)

    def learn(self):
        alpha = self.network.log_alpha.exp().detach()
        states, actions, rewards, next_states, dones = self.memory.sample()
        dones = dones.unsqueeze(-1)
        rewards = rewards.unsqueeze(-1)
        # Compute target Q and Target V
        next_targeted_value = self.network.value_target(states)
        target_q = (rewards + self.GAMMA * (1 - dones) * next_targeted_value).detach()
        assert next_targeted_value.shape == dones.shape, " next_targeted_value {} != dones {}".format(next_targeted_value.shape, dones.shape)
        assert next_targeted_value.shape == rewards.shape, " next_targeted_value {} != rewards {}".format(next_targeted_value.shape, rewards.shape)

        new_actions, new_log_probs = self.network.get_with_probabilities(states)
        concatenated = torch.cat((states, new_actions), dim=-1)
        current_q = torch.stack([critic(concatenated) for critic in self.network.critics_local])
        min_current_q, _ = current_q.min(dim=0)
        assert min_current_q.shape == new_log_probs.shape, " min_current_q {} != new_log_probs {}".format(
            min_current_q.shape, new_log_probs.shape)
        self.estimation = min_current_q.mean().detach().cpu().numpy().item()
        target_v = (min_current_q - alpha * new_log_probs).detach()

        for optim, critic, i in zip(self.critic_optims, self.network.critics_local,
                                    range(len(self.network.critics_local))):
            concatenated = torch.cat((states, actions), dim=-1)
            q_value = critic(concatenated)
            optim.zero_grad()
            loss = self.loss_function(q_value, target_q)
            loss.backward()
            self.critics_losses[i] = loss.detach().cpu().numpy().item()
            optim.step()

        self.value_optim.zero_grad()
        self.loss_function(self.network.value_local(states), target_v).backward()
        self.value_optim.step()

        # optimize alpha
        alpha_loss = - self.network.log_alpha * (new_log_probs + self.TARGET_ENTROPY).detach()
        self.alpha_optim.zero_grad()
        alpha_loss.mean().backward()
        self.alpha_optim.step()

        # optimize the actor
        policy_kl_losses = - (min_current_q - alpha * new_log_probs)
        self.actor_optim.zero_grad()
        loss = policy_kl_losses.mean()
        loss.backward()
        self.policy_loss = loss.detach().cpu().numpy().item()
        self.actor_optim.step()
        self.update_target()
