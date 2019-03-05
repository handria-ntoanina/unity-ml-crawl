import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
from agents_maddpg.utils import soft_update
import numpy as np


class MADDPG():
    """Interacts with and learns from the environment."""
    
    def __init__(self, network_local, network_target, num_agents, memory,device,
                GRADIENT_CLIP,
                BOOTSTRAP_SIZE,
                GAMMA, 
                TAU, 
                LR_CRITIC,
                LR_ACTOR, 
                UPDATE_EVERY,
                TRANSFER_EVERY,
                UPDATE_LOOP,
                ADD_NOISE_EVERY,
                WEIGHT_DECAY,
                MEMORY_RANDOMNESS):
        
        # Actor networks
        self.network_local = network_local
        self.network_target = network_target

        self.set_optimizer_hyperparameters(LR_CRITIC, LR_ACTOR, WEIGHT_DECAY)
        
        self.device = device
        
        # Noise
        self.noise = None
        
        # Replay memory
        self.memory = memory
        # Initialize time steps (for updating every UPDATE_EVERY steps)
        self.u_step = 0
        self.n_step = 0

        self.TAU = TAU
        self.UPDATE_EVERY = UPDATE_EVERY
        self.TRANSFER_EVERY = TRANSFER_EVERY
        self.UPDATE_LOOP = UPDATE_LOOP
        self.ADD_NOISE_EVERY = ADD_NOISE_EVERY
        self.GRADIENT_CLIP = GRADIENT_CLIP
        self.MEMORY_RANDOMNESS = MEMORY_RANDOMNESS

        self.set_bootstrap_size(BOOTSTRAP_SIZE, GAMMA, num_agents)

        self.loss_function = torch.nn.SmoothL1Loss(reduce=None)

    def set_isw_impact(self, value):
        self.ISW_IMPACT = value

    def set_isw_impact_increment(self, value):
        self.ISW_IMPACT_INCREMENT = value

    def set_bootstrap_size(self, BOOTSTRAP_SIZE, GAMMA, num_agents):
        # initialize these variables to store the information of the n-previous timestep that are necessary to apply the bootstrap_size
        self.BOOTSTRAP_SIZE = BOOTSTRAP_SIZE
        self.GAMMA = GAMMA
        self.rewards = deque(maxlen=BOOTSTRAP_SIZE)
        self.states = deque(maxlen=BOOTSTRAP_SIZE)
        self.actions = deque(maxlen=BOOTSTRAP_SIZE)
        self.gammas = np.array([[GAMMA ** i for j in range(num_agents)] for i in range(BOOTSTRAP_SIZE)])

    def set_optimizer_hyperparameters(self, LR_CRITIC, LR_ACTOR, WEIGHT_DECAY):
        self.actor_optim = optim.Adam(self.network_local.actor.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        self.critic_optim = optim.Adam(self.network_local.critic.parameters(), lr=LR_ACTOR, weight_decay=WEIGHT_DECAY)
        self.WEIGHT_DECAY = WEIGHT_DECAY
        self.LR_CRITIC = LR_CRITIC
        self.LR_ACTOR = LR_ACTOR

    def reset(self):
        if self.noise:
            for n in self.noise:
                n.reset()
        
    def set_noise(self, noise):
        self.noise = noise
    
    def act(self, states, noise = 0.0):
        """Returns actions of each actor for given states.
        
        Params
        ======
            state (array_like): current states
            add_noise: either alter the decision of the actor network or not. During training, this is necessary to promote the exploration. However, during validation, this is altering the agent and should be deactivated.
        """
        ret = None
        
        self.n_step = (self.n_step + 1) % self.ADD_NOISE_EVERY
        
        with torch.no_grad():
            self.network_local.eval()
            states = torch.from_numpy(states).float().unsqueeze(0).to(self.device)
            ret = self.network_local(states).squeeze().cpu().data.numpy()
            self.network_local.train()
            if self.n_step == 0:
                for i in range(len(ret)):
                    ret[i] += noise * self.noise[i].sample()
        return ret
    
    def step(self, states, actions, rewards, next_states, dones):
        # Save experience in replay memory
        
        self.rewards.append(rewards)
        self.states.append(states)
        self.actions.append(actions)
            
        if len(self.rewards) == self.BOOTSTRAP_SIZE:
            # get the sum of rewards per agents
            reward = np.sum(self.rewards * self.gammas, axis = -2)

            with torch.no_grad():
                self.network_target.eval()
                next_states_tensor = torch.tensor(next_states, device=self.device, dtype=torch.float32, requires_grad=False)
                dones_tensor = torch.tensor(dones, device=self.device, dtype=torch.float32, requires_grad=False)
                reward_tensor = torch.tensor(reward, device=self.device, dtype=torch.float32, requires_grad=False)
                next_actions = self.network_target(next_states_tensor)
                q_next = self.network_target.estimate(next_states_tensor, next_actions).squeeze(-1)
                assert q_next.shape == dones_tensor.shape, " q_next {} != dones {}".format(q_next.shape, dones_tensor.shape)
                assert q_next.shape == reward_tensor.shape, " q_next {} != rewards {}".format(q_next.shape, reward_tensor.shape)
                targeted_value = reward_tensor + (self.GAMMA ** self.BOOTSTRAP_SIZE) * q_next * (1 - dones_tensor)

                states_tensor = torch.tensor(self.states[0], device=self.device, dtype=torch.float32, requires_grad=False)
                actions_tensor = torch.tensor(self.actions[0], device=self.device, dtype=torch.float32, requires_grad=False)
                current_value = self.network_local.estimate(states_tensor, actions_tensor).squeeze(-1)
                assert targeted_value.shape == current_value.shape, " targeted_value {} != current_value {}".format(
                    targeted_value.shape, current_value.shape)

            errors = torch.abs(current_value - targeted_value).data.numpy() + 0.01
            for i in range(len(errors)):
                self.memory.add(errors[i] ** self.MEMORY_RANDOMNESS, self.states[0][i],
                                            self.actions[0][i],
                                            reward[i],
                                            next_states[i],
                                            dones[i])
            
        if np.any(dones):
            self.rewards.clear()
            self.states.clear()
            self.actions.clear()
            
        # Learn every UPDATE_EVERY timesteps
        self.u_step = (self.u_step + 1) % self.UPDATE_EVERY

        if len(self.memory) > self.memory.batch_size and self.u_step == 0:
            t_step=0
            for _ in range(self.UPDATE_LOOP):
                self.learn()
                self.ISW_IMPACT = min(1, self.ISW_IMPACT + self.ISW_IMPACT_INCREMENT)
                t_step=(t_step + 1) % self.TRANSFER_EVERY
                if t_step == 0:
                    soft_update(self.network_local, self.network_target, self.TAU)
    
    def learn(self):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
        """
        # sample the memory to disrupt the internal correlation
        states, actions, rewards, next_states, dones, idx, sampling_probabilities = self.memory.sample()

        # The critic should estimate the value of the states to be equal to rewards plus
        # the estimation of the next_states value according to the critic_target and actor_target
        with torch.no_grad():
            self.network_target.eval()
            next_actions = self.network_target(next_states)
            # the rewards here was pulled from the memory. Before being registered there, the rewards are already considering the size of the bootstrap with the appropriate discount factor
            q_next = self.network_target.estimate(next_states, next_actions).squeeze(-1)
            assert q_next.shape == dones.shape, " q_next {} != dones {}".format(q_next.shape, dones.shape)
            assert q_next.shape == rewards.shape, " q_next {} != rewards {}".format(q_next.shape, rewards.shape)
            # TODO this might need to be detached
            targeted_value = rewards + (self.GAMMA**self.BOOTSTRAP_SIZE)*q_next*(1 - dones)
         
        current_value = self.network_local.estimate(states, actions).squeeze(-1)
        assert targeted_value.shape == current_value.shape, " targeted_value {} != current_value {}".format(targeted_value.shape, current_value.shape)

        # update the probabilities
        errors = (torch.abs(current_value - targeted_value) + 0.01).data.numpy() ** self.MEMORY_RANDOMNESS
        for i in range(len(idx)):
            self.memory.update(idx[i], errors[i])
        # calculate the loss of the critic network and backpropagate
        self.critic_optim.zero_grad()
        loss = self.loss_function(current_value, targeted_value) * 1 / (len(self.memory) * sampling_probabilities).pow(self.ISW_IMPACT)
        loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(self.network_local.critic.parameters(), self.GRADIENT_CLIP)
        self.critic_optim.step()


        # optimize the actor by having the critic evaluating the value of the actor's decision
        self.actor_optim.zero_grad()
        actions_pred = self.network_local(states)
        mean = self.network_local.estimate(states, actions_pred).mean()
        # during the back propagation, parameters of the actor that led to a bad note from the critic will be demoted, and good parameters that led to a good note will be promoted
        (-mean).backward()
        self.actor_optim.step()    