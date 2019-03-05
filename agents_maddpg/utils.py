import numpy as np
import random
import copy
from collections import namedtuple, deque
import torch
import time
import random
import numpy as np
from agents_maddpg.sumtree import SumTree

def soft_update(local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    θ_target = θ_target + τ*(θ_local - θ_target)
    θ_local = r + gamma * θ_local(s+1)

    Params
    ======
        local_model (PyTorch model): weights will be copied from
        target_model (PyTorch model): weights will be copied to
        tau (float): interpolation parameter 
    """
    # this is transferring gradually the parameters of the online Q Network to the fixed one
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class SimpleNoise:
    def __init__(self, size, scale = 2.0):
        self.size = size
        self.scale = scale
        self.median = scale / 2
        
    def reset(self):
        pass
    
    def sample(self):
        return self.scale * np.random.randn(self.size) - self.median
        
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
    
Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, device, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
       
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.device = device
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        return self.to_tensor(experiences)
    
    def shuffle_all(self):
        temp = list(self.memory)
        random.shuffle(temp)
        batch_count = int(len(temp) / self.batch_size)
        for a in range(batch_count):
            yield self.to_tensor(temp[a:(a+1)*self.batch_size])
    
    def to_tensor(self, experiences):
        states = torch.from_numpy(np.array([e.state for e in experiences if e is not None], dtype=np.float32)).to(self.device).requires_grad_(False)
        actions = torch.from_numpy(np.array([e.action for e in experiences if e is not None], dtype=np.float32)).to(self.device).requires_grad_(False)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences if e is not None], dtype=np.float32)).to(self.device).requires_grad_(False)
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences if e is not None], dtype=np.float32)).to(self.device).requires_grad_(False)
        dones = torch.from_numpy(np.array([e.done for e in experiences if e is not None], dtype=np.float32)).to(self.device).requires_grad_(False)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class PrioritizedMemory:
    def __init__(self, device, capacity, batch_size):
        self.tree = None
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.reset()
        
    def reset(self):
#         if self.tree:
#             del self.tree
        self.tree = SumTree(self.capacity)
        


    def add(self, error, state, action, reward, next_state, done):
        # print("error={}, state={}, action={}, reward={}, next_state={}, done={}".format(error, state.shape,
        #                                                                                 action.shape, reward, next_state.shape, done))
        sample = Experience(state, action, reward, next_state, done)
        self.tree.add(error, sample)

    def sample(self):
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        idxs = []
        segment = self.tree.total() / self.batch_size
        priorities = []

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, e) = self.tree.get(s)
            priorities.append(p)
            states.append(e.state)
            actions.append(e.action)
            rewards.append(e.reward)
            next_states.append(e.next_state)
            dones.append(e.done)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        states = torch.tensor(states, device=self.device, dtype=torch.float32, requires_grad=False)
        actions = torch.tensor(actions, device=self.device, dtype=torch.float32, requires_grad=False)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32, requires_grad=False)
        next_states = torch.tensor(next_states, device=self.device, dtype=torch.float32, requires_grad=False)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float32, requires_grad=False)
        sampling_probabilities = torch.tensor(sampling_probabilities, device=self.device, dtype=torch.float32, requires_grad=False)

        return states, actions, rewards, next_states, dones, idxs, sampling_probabilities

    def update(self, idx, error):
        self.tree.update(idx, error)

    def __len__(self):
        return self.tree.n_entries
