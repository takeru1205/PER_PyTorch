from collections import deque
import gym
import numpy as np
import torch
from dqn_torch import Agent
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='./logs')
env = gym.make('CartPole-v1')
SEED = 42
torch.manual_seed(SEED)
env.seed(SEED)
np.random.seed(SEED)
agent = Agent(env=env, writer=writer)
last_5_episode_scores = deque(maxlen=5)

epoch = 300
max_episode = 200

steps = 0
for e in range(epoch):
    state = env.reset()
    done = False
    timestep = 0
    cumulative_reward = 0
    while not done:
        action = agent.choose_action(state, steps)
        state_, reward, done, _ = env.step(action)
        # env.render()
        cumulative_reward += 1
        agent.store_transition(state, action, reward, state_, done)
        agent.model_update(steps)

        timestep += 1
        steps += 1
        if timestep == max_episode:
            break
        state = state_
    if e % 5 == 0:
        agent.target_update()
    writer.add_scalar("score", cumulative_reward, e)
    print('Episode {}: / Reward: {}'.format(e, cumulative_reward))
    last_5_episode_scores.append(cumulative_reward)
    if sum(list(last_5_episode_scores)) // 5 == 200:
        break
