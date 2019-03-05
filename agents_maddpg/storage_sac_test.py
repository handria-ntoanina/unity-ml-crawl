from agents_maddpg.model import TanhGaussianActorCritic
from agents_maddpg.storage_sac import Storage
import torch.nn.functional as F

state_size = 115
action_size = 114
device = 'cpu'
memory_size = int(1e5)
memory_size=memory_size
batch_size=16
ACTIVATION = F.leaky_relu
TAU=1e-2
LR_CRITIC = 1e-4
LR_ACTOR = 1e-5
LR_ALPHA = 1e-6
UPDATE_EVERY=2
TRANSFER_EVERY=1
UPDATE_LOOP=3
GAMMA=0.992
TARGET_ENTROPY=6
Q_NUMBER = 1
WEIGHT_DECAY = 0
initial = Storage.new( TanhGaussianActorCritic, state_size, action_size, device,
                    memory_size=memory_size,
                    batch_size=batch_size,
                    ACTIVATION = ACTIVATION,
                    TAU=TAU,
                    LR_CRITIC = LR_CRITIC,
                    LR_ACTOR = LR_ACTOR,
                    LR_ALPHA = LR_ALPHA,
                    UPDATE_EVERY=UPDATE_EVERY,
                    TRANSFER_EVERY=TRANSFER_EVERY,
                    UPDATE_LOOP=UPDATE_LOOP,
                    GAMMA=GAMMA,
                    TARGET_ENTROPY=TARGET_ENTROPY,
                    Q_NUMBER = Q_NUMBER,
                    WEIGHT_DECAY = WEIGHT_DECAY)
Storage.save('temp.cpk', [], initial)
agent, scores = Storage.load('temp.cpk', device)
assert state_size == agent.network.actor.fc.hidden_layers[0].in_features
for critic in agent.network.critics_local:
    assert state_size + action_size == critic.hidden_layers[0].in_features
    assert ACTIVATION == critic.activation
for critic in agent.network.critics_target:
    assert state_size + action_size == critic.hidden_layers[0].in_features
    assert ACTIVATION == critic.activation

assert TAU == agent.TAU
assert LR_CRITIC == agent.LR_CRITIC
assert LR_ACTOR == agent.LR_ACTOR
assert LR_ALPHA == agent.LR_ALPHA
assert UPDATE_EVERY == agent.UPDATE_EVERY
assert TRANSFER_EVERY == agent.TRANSFER_EVERY
assert UPDATE_LOOP == agent.UPDATE_LOOP
assert GAMMA == agent.GAMMA
assert TARGET_ENTROPY == agent.TARGET_ENTROPY
assert Q_NUMBER == len(agent.network.critics_target)
assert WEIGHT_DECAY == agent.WEIGHT_DECAY
print("Hyper parameters checked!")

for a, b in zip(agent.network.parameters(), initial.network.parameters()):
    sum_a = a.sum()
    sum_b = b.sum()
    assert sum_a == sum_b
print("Model parameters checked!")

for a, b in zip(agent.network.actor.parameters(), initial.network.actor.parameters()):
    sum_a = a.sum()
    sum_b = b.sum()
    assert sum_a == sum_b
print("Actor parameters checked!")

for a, b in zip(agent.actor_optim.param_groups[0]['params'], initial.actor_optim.param_groups[0]['params']):
    sum_a = a.sum()
    sum_b = b.sum()
    assert sum_a == sum_b , "{} != {}".format(sum_a, sum_b)
print("Actor optimizer checked!")


for c, d in zip(agent.network.critics_local, initial.network.critics_local):
    sum_a = 0
    sum_b = 0
    for a, b in zip(c.parameters(), d.parameters()):
        sum_a += a.sum()
        sum_b += b.sum()
    assert sum_a == sum_b , "{} != {}".format(sum_a, sum_b)
print("Critic local parameters checked!")

for c, d in zip(agent.network.critics_local, initial.network.critics_target):
    sum_a = 0
    sum_b = 0
    for a, b in zip(c.parameters(), d.parameters()):
        sum_a += a.sum()
        sum_b += b.sum()
    assert sum_a == sum_b , "{} != {}".format(sum_a, sum_b)
print("Critic target parameters checked!")

for c, d in zip(agent.critic_optims, initial.critic_optims):
    for a, b in zip(c.param_groups[0]['params'], d.param_groups[0]['params']):
        sum_a = a.sum()
        sum_b = b.sum()
        assert sum_a == sum_b , "{} != {}".format(sum_a, sum_b)
print("Critics optimizer checked!")