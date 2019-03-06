import torch
import torch.nn.functional as F
from agents_maddpg.sac import SAC
from agents_maddpg.sac_value import SAC_Value
from agents_maddpg.utils import ReplayBuffer
from shutil import copyfile
import agents_maddpg
import os.path


class Storage:
    def __init__(self):
        pass

    @staticmethod
    def new(network, state_size, action_size, device, memory_size=int(1e5),
            batch_size=256,
            ACTIVATION = F.relu,
            TAU = 1e-3,
            LR_CRITIC = 5e-4,
            LR_ACTOR = 5e-4,
            LR_ALPHA = 5e-4,
            UPDATE_EVERY = 1,
            TRANSFER_EVERY = 2,
            UPDATE_LOOP = 10,
            WEIGHT_DECAY = 0,
            TARGET_ENTROPY=0.25,
            GAMMA=0.99,
            Q_NUMBER = 2 ):

        network = network(state_size, action_size, Q_NUMBER, activation = ACTIVATION).to(device)

        memory = ReplayBuffer(device, memory_size, batch_size)
        agent = SAC(network, memory, device,
                       LR_CRITIC,
                       LR_ACTOR, LR_ALPHA, UPDATE_EVERY,
                        UPDATE_LOOP, TAU, TRANSFER_EVERY,
                       WEIGHT_DECAY, TARGET_ENTROPY, GAMMA)
        agent.update_target(tau = 1)

        return agent

    @staticmethod
    def new_sac_value(network, state_size, action_size, device, memory_size=int(1e5),
            batch_size=128,
            ACTIVATION = F.leaky_relu,
            TAU = 1e-2,
            LR_CRITIC = 3e-4,
            LR_ACTOR = 3e-4,
            LR_ALPHA = 3e-4,
            UPDATE_EVERY = 1,
            TRANSFER_EVERY = 1,
            UPDATE_LOOP = 1,
            WEIGHT_DECAY = 0,
            TARGET_ENTROPY=1.2,
            GAMMA=0.99,
            Q_NUMBER = 2 ):

        network = network(state_size, action_size, Q_NUMBER, activation = ACTIVATION).to(device)

        memory = ReplayBuffer(device, memory_size, batch_size)
        agent = SAC_Value(network, memory, device,
                       LR_CRITIC,
                       LR_ACTOR, LR_ALPHA, UPDATE_EVERY,
                        UPDATE_LOOP, TAU, TRANSFER_EVERY,
                       WEIGHT_DECAY, TARGET_ENTROPY, GAMMA)
        agent.update_target(tau = 1)

        return agent

    @staticmethod
    def save(filename, scores, agent: SAC):
        checkpoint = {'activation': 'torch.nn.functional.' + agent.network.activation.__name__,
                      'state_size': agent.network.state_size,
                      'action_size': agent.network.action_size,
                      'network': 'agents_maddpg.model.' + agent.network.__class__.__name__,
                      'q_number': len(agent.network.critics_local),
                      'network_params': agent.network.state_dict(),
                      'actor_params': agent.network.actor.state_dict(),
                      'critics_local_params':  [critic.state_dict() for critic in agent.network.critics_local],
                      'critics_target_params':  [critic.state_dict() for critic in agent.network.critics_target],
                      'TAU': agent.TAU,
                      'LR_CRITIC': agent.LR_CRITIC,
                      'LR_ACTOR': agent.LR_ACTOR,
                      'LR_ALPHA': agent.LR_ALPHA,
                      'UPDATE_EVERY': agent.UPDATE_EVERY,
                      'TRANSFER_EVERY': agent.TRANSFER_EVERY,
                      'UPDATE_LOOP': agent.UPDATE_LOOP,
                      'WEIGHT_DECAY': agent.WEIGHT_DECAY,
                      'TARGET_ENTROPY': agent.TARGET_ENTROPY,
                      'GAMMA': agent.GAMMA,
                      'scores': scores,
                      'actor_optim': agent.actor_optim.state_dict(),
                      'critic_optims': [optim.state_dict() for optim in agent.critic_optims],
                      'alpha_optim':agent.alpha_optim.state_dict()}
        torch.save(checkpoint, filename)
        folder = '.\\' + '\\'.join(filename.split('\\')[0:-1])
        try:
            torch.save(agent.memory, folder + '\memory.mem')
            copyfile(folder + '\memory.mem', folder + '\memory_complete.mem')
        except:
            pass

    @staticmethod
    def load(filename, device, agent_class=SAC):
        checkpoint = torch.load(filename)
        folder = './' + '/'.join(filename.split('/')[0:-1])
        pathMemory = folder + '/memory_complete.mem'
        if os.path.isfile(pathMemory):
            memory = torch.load(pathMemory)
        else:
            memory = ReplayBuffer(device, int(1e5), 16)
        activation = eval(checkpoint['activation'])
        network = eval(checkpoint['network'])(checkpoint['state_size'], checkpoint['action_size'],
                                              checkpoint['q_number'], activation=activation).to(device)
        network.load_state_dict(checkpoint['network_params'])
        # if 'actor_params' in checkpoint:
        network.actor.load_state_dict(checkpoint['actor_params'])
        # if 'critics_local_params' in checkpoint:
        for state_dict, critic in zip(checkpoint['critics_local_params'], network.critics_local):
            critic.load_state_dict(state_dict)
        # if 'critics_target_params' in checkpoint:
        for state_dict, critic in zip(checkpoint['critics_target_params'], network.critics_target):
            critic.load_state_dict(state_dict)

        agent = agent_class(network,memory,device, checkpoint['LR_CRITIC'],
                    checkpoint['LR_ACTOR'], checkpoint['LR_ALPHA'], checkpoint['UPDATE_EVERY'],
                    checkpoint['UPDATE_LOOP'], checkpoint['TAU'],
                    checkpoint['TRANSFER_EVERY'],
                    checkpoint['WEIGHT_DECAY'], checkpoint['TARGET_ENTROPY'],
                    checkpoint['GAMMA'])
        # if 'alpha_optim' in checkpoint:
        agent.alpha_optim.load_state_dict(checkpoint['alpha_optim'])
        # if 'critic_optims' in checkpoint:
        for state, optim in zip(checkpoint['critic_optims'], agent.critic_optims):
            optim.load_state_dict(state)
        # if 'actor_optim' in checkpoint:
        agent.actor_optim.load_state_dict(checkpoint['actor_optim'])
        scores = checkpoint['scores']
        return agent, scores
