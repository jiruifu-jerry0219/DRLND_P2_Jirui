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
env = UnityEnvironment(file_name='Reacher_Linux/Reacher.x86_64')

# select this option to load version 2 (with 20 agents) of the environment
# env = UnityEnvironment(file_name='/data/Reacher_Linux_NoVis/Reacher.x86_64')


# Environments contain **_brains_** which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, and set it as the default brain we will be controlling from Python.

# In[2]:


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


# ### 3. Take Random Actions in the Environment
#
# In the next code cell, you will learn how to use the Python API to control the agent and receive feedback from the environment.
#
# Note that **in this coding environment, you will not be able to watch the agents while they are training**, and you should set `train_mode=True` to restart the environment.

# In[5]:


# env_info = env.reset(train_mode=True)[brain_name]      # reset the environment
# states = env_info.vector_observations                  # get the current state (for each agent)
# scores = np.zeros(num_agents)                          # initialize the score (for each agent)
# while True:
#     actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
#     actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
#     env_info = env.step(actions)[brain_name]           # send all actions to tne environment
#     next_states = env_info.vector_observations         # get next state (for each agent)
#     rewards = env_info.rewards                         # get reward (for each agent)
#     dones = env_info.local_done                        # see if episode finished
#     scores += env_info.rewards                         # update the score (for each agent)
#     states = next_states                               # roll over states to next time step
#     if np.any(dones):                                  # exit loop if episode finished
#         break
# print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))


# When finished, you can close the environment.

# In[6]:


# env.close()


# ### 4. It's Your Turn!
#
# Now it's your turn to train your own agent to solve the environment!  A few **important notes**:
# - When training the environment, set `train_mode=True`, so that the line for resetting the environment looks like the following:
# ```python
# env_info = env.reset(train_mode=True)[brain_name]
# ```
# - To structure your work, you're welcome to work directly in this Jupyter notebook, or you might like to start over with a new file!  You can see the list of files in the workspace by clicking on **_Jupyter_** in the top left corner of the notebook.
# - In this coding environment, you will not be able to watch the agents while they are training.  However, **_after training the agents_**, you can download the saved model weights to watch the agents on your own machine!

# #### 4.1 Training with DDPG

# In[4]:


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


n_episodes = 2000
max_t = 1000
print_every = 100
min_mean_score = 30

scores_deque = deque(maxlen = print_every)
scores = []
max_score = -np.Inf
for i_episode in range(1, n_episodes + 1):
    env_info = env.reset(train_mode = True)[brain_name]
    state = env_info.vector_observations[0]

    agent.reset()
    score = 0
    for t in range(max_t):
        action = agent.act(state)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]

        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward

        if done:
            break

    scores_deque.append(score)
    scores.append(score)
    print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_deque), score), end="")

    if i_episode % print_every == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_deque)))
    if len(scores_deque) == print_every:
        mean_score = np.mean(scores_deque)
        if mean_score >= min_mean_score:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            print('\rCheckpoint saved under average score: {:.2f}'.format(min_mean_score))
    # print('\rTime consumed: {:.2f}'.format(time1 - time0))





# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores) + 1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
plt.savefig("ResultDDPG.png")


# In[ ]:
