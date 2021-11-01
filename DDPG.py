#!/usr/bin/env python
# coding: utf-8

# # Continuous Control
#
# ---
#
# You are welcome to use this coding environment to train your agent for the project.  Follow the instructions below to get started!
#
# ### 1. Start the Environment
#
# Run the next code cell to install a few packages.  This line will take a few minutes to run!

# The environments corresponding to both versions of the environment are already saved in the Workspace and can be accessed at the file paths provided below.
#
# Please select one of the two options below for loading the environment.

# In[1]:


from unityagents import UnityEnvironment
import numpy as np

# select this option to load version 1 (with a single agent) of the environment
# env = UnityEnvironment(file_name='Reacher_Linux/Reacher.x86_64')
# env = UnityEnvironment(file_name='Reacher_One_Linux_NoVis/Reacher_One_Linux_NoVis.x86_64')
env = UnityEnvironment(file_name = 'Reacher_Linux_NoVis/Reacher.x86_64')




# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# ### 2. Examine the State and Action Spaces
#
# Run the code cell below to print some information about the environment.

# In[3]:


# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])





import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

from ddpg_agent import Agent
from time import time


# In[5]:


agent = Agent(state_size, action_size, random_seed=0)


# In[6]:


n_episodes = 1000
max_t = 1001
print_every = 100
min_mean_score = 30

scores_deque = deque(maxlen = print_every)
scores = []
max_score = -np.Inf
for i_episode in range(1, n_episodes + 1):
    env_info = env.reset(train_mode = True)[brain_name]
    state = env_info.vector_observations


    agent.reset()
    score = np.zeros(num_agents)
    for t in range(max_t):
        action = agent.act(state)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations
        reward = env_info.rewards
        done = env_info.local_done

        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward

        if np.any(done): #for multiagent case, if single agent just use "if done:"
            break

    scores_deque.append(np.mean(score))
    scores.append(np.mean(score))
    print('\rEpisode {}\tScore: {:.2f}'.format(i_episode, np.mean(score)), end="")

    if i_episode % print_every == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
    if np.mean(scores_deque) >= min_mean_score:
        print('\nEnvironment solved in {} episodes', format(i_episode))
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')



fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores) + 1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
plt.savefig("ResultDDPG.png")


# In[ ]:
