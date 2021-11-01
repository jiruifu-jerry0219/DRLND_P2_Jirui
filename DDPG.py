


from unityagents import UnityEnvironment
import numpy as np

"""
select this option to load version 1 (with a single agent) of the environment
"""

# env = UnityEnvironment(file_name='Reacher_Linux/Reacher.x86_64')
# env = UnityEnvironment(file_name='Reacher_One_Linux_NoVis/Reacher_One_Linux_NoVis.x86_64')
env = UnityEnvironment(file_name = 'Reacher_Linux_NoVis/Reacher.x86_64')




"""
get the default brain
"""

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
import gym





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
    state = env_info.vector_observations

    state = env.reset()

    agent.reset()
    # score = np.zeros(num_agents)
    score = np.zeros(1)

    for t in range(max_t):
        action = agent.act(state)

        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations

        rewards = env_info.rewards

        done = env_info.local_done

        score += rewards

        agent.step(state, action, rewards, next_state, done, t)

        state = next_state

        if np.any(done): #for multiagent case, if single agent just use "if done:"
            break
    scores_deque.append(np.mean(score))
    scores.append(np.mean(score))

    print('\rEpisode {}\tThe average score of this episode: {:.2f}'.format(i_episode, np.mean(score)), end="")
    torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
    torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')

    if i_episode % print_every == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

    if np.mean(scores[-100:]) >= min_mean_score:
        print('\nEnvironment solved in {} episodes', format(i_episode))
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        break



import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores) + 1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
plt.savefig("ResultDDPG.png")
