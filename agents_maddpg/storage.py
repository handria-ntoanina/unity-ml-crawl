import torch
import torch.nn.functional as F
from agents_maddpg.maddpg import MADDPG
from agents_maddpg.utils import PrioritizedMemory
from agents_maddpg.utils import soft_update
from shutil import copyfile
import agents_maddpg


class Storage:
    def __init__(self):
        pass

    @staticmethod
    def new(network, state_size, action_size, num_agents,device, noise, seed, memory_size=int(1e5),
            batch_size=256,
            ACTIVATION = F.relu,
            GRADIENT_CLIP = 1,
            BOOTSTRAP_SIZE = 5,
            GAMMA = 0.99,
            TAU = 1e-3,
            LR_CRITIC = 5e-4,
            LR_ACTOR = 5e-4,
            UPDATE_EVERY = 1,
            TRANSFER_EVERY = 2,
            UPDATE_LOOP = 10,
            ADD_NOISE_EVERY = 5,
            WEIGHT_DECAY = 0,
            MEMORY_RANDOMNESS = 1):

        network_local = network(state_size, action_size, activation = ACTIVATION).to(device)
        network_target = network(state_size, action_size, activation = ACTIVATION).to(device)
        soft_update(network_local, network_target, 1)

        memory = PrioritizedMemory(device, memory_size, batch_size)
        agent = MADDPG(network_local, network_target, num_agents, memory, device, GRADIENT_CLIP,
                       BOOTSTRAP_SIZE, GAMMA, TAU, LR_CRITIC,
                       LR_ACTOR, UPDATE_EVERY,
                       TRANSFER_EVERY, UPDATE_LOOP, ADD_NOISE_EVERY,
                       WEIGHT_DECAY, MEMORY_RANDOMNESS)
        noises = [noise(action_size) for i in range(num_agents)]
        agent.set_noise(noises)
        agent.set_isw_impact(1e-5)
        agent.set_isw_impact_increment(1e-5)

        return agent

    @staticmethod
    def save(filename, scores, agent: MADDPG):
        checkpoint = {'activation': 'torch.nn.functional.' + agent.network_local.activation.__name__,
                      'noise': 'agents_maddpg.utils.' + agent.noise[0].__class__.__name__,
                      'state_size': agent.network_local.state_size,
                      'action_size': agent.network_local.action_size,
                      'network': 'agents_maddpg.model.' + agent.network_local.__class__.__name__,
                      'network_local': agent.network_local.state_dict(),
                      'network_target': agent.network_target.state_dict(),
                      'GRADIENT_CLIP': agent.GRADIENT_CLIP,
                      'BOOTSTRAP_SIZE': agent.BOOTSTRAP_SIZE,
                      'GAMMA': agent.GAMMA,
                      'TAU': agent.TAU,
                      'LR_CRITIC': agent.LR_CRITIC,
                      'LR_ACTOR': agent.LR_ACTOR,
                      'UPDATE_EVERY': agent.UPDATE_EVERY,
                      'TRANSFER_EVERY': agent.TRANSFER_EVERY,
                      'UPDATE_LOOP': agent.UPDATE_LOOP,
                      'ADD_NOISE_EVERY': agent.ADD_NOISE_EVERY,
                      'WEIGHT_DECAY': agent.WEIGHT_DECAY,
                      'MEMORY_RANDOMNESS': agent.MEMORY_RANDOMNESS,
                      'scores': scores,
                      'ISW_IMPACT' : agent.ISW_IMPACT,
                      'ISW_IMPACT_INCREMENT' : agent.ISW_IMPACT_INCREMENT}
        torch.save(checkpoint, filename)
        folder = '.\\' + '\\'.join(filename.split('\\')[0:-1])
        torch.save(agent.memory, folder + '\memory.mem')
        copyfile(folder + '\memory.mem', folder + '\memory_complete.mem')

    @staticmethod
    def load(filename, device, num_agents) -> MADDPG:
        checkpoint = torch.load(filename)
        folder = '.\\' + '\\'.join(filename.split('\\')[0:-1])
        memory = torch.load(folder + '\memory_complete.mem')
        activation = eval(checkpoint['activation'])
        network_local = eval(checkpoint['network'])(checkpoint['state_size'],checkpoint['action_size'], activation=activation).to(device)
        network_target = eval(checkpoint['network'])(checkpoint['state_size'],checkpoint['action_size'], activation=activation).to(device)
        network_local.load_state_dict(checkpoint['network_local'])
        network_target.load_state_dict(checkpoint['network_target'])
        agent = MADDPG(network_local,network_target,num_agents,memory,device, checkpoint['GRADIENT_CLIP'],
                       checkpoint['BOOTSTRAP_SIZE'], checkpoint['GAMMA'], checkpoint['TAU'],checkpoint['LR_CRITIC'],
                       checkpoint['LR_ACTOR'], checkpoint['UPDATE_EVERY'],
                       checkpoint['TRANSFER_EVERY'], checkpoint['UPDATE_LOOP'], checkpoint['ADD_NOISE_EVERY'],
                       checkpoint['WEIGHT_DECAY'], checkpoint['MEMORY_RANDOMNESS'])
        noise = eval(checkpoint['noise'])
        noises = [noise(checkpoint['action_size']) for i in range(num_agents)]
        agent.set_noise(noises)
        agent.set_isw_impact(checkpoint['ISW_IMPACT'])
        agent.set_isw_impact_increment(checkpoint['ISW_IMPACT_INCREMENT'])

        scores = checkpoint['scores']
        return agent, scores
