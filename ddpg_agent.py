import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

"""
Define the basic parameters of reinforcement learning
BUFFER_SIZE: replay buffer Size
BATCH_SIZE: minibatch size for experience replay
GAMMA: discount factor
TAU: eligibility trace for updating the target
LR_ACTOR: learning rate for actor network
LR_CRITIC: learning rate for cricit network
(what's this?) WEIGHT_DECAY: L2 weight decay
"""
BUFFER_SIZE = int(1e9)
BATCH_SIZE = 512
GAMMA = 0.99
TAU = 1e-3
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
WEIGHT_DECAY = 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Current device is:', device)
print('The GPU model is: ', torch.cuda.get_device_name(0))

class Agent():
    def __init__(self, state_size, action_size, random_seed):
        """Initialize the DDPG Agent
        parameters
        ====
        state_size (int):dimension of state space
        action_size (int): dimension of action space
        random_seed(int):random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.random_seed = random_seed

        # Define the actor network
        self.actor_local = Actor(self.state_size,
                                self.action_size,
                                self.random_seed,
                                hidden_dims = (256, 126, 64, 32)).to(device)
        self.actor_target = Actor(self.state_size,
                                self.action_size,
                                self.random_seed,
                                hidden_dims = (256, 126,64,  32)).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr = LR_ACTOR)

        # Define the critic network
        self.critic_local = Critic(self.state_size,
                                action_size = self.action_size,
                                seed = self.random_seed,
                                fcs_size = 400,
                                hidden_dims = (256, 126, 64, 32)).to(device)
        self.critic_target = Critic(self.state_size,
                                action_size = self.action_size,
                                seed = self.random_seed,
                                fcs_size = 400,
                                hidden_dims = (256, 126,64,  32)).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr = LR_CRITIC)

        # Random noise for exploration
        self.noise = OUNoise(action_size, random_seed)

        # Replay BUFFER
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def step(self, state, action, reward, next_state, done):
        """Save experience to replay buffer then pick a previous experience
        from the replay buffer to learn"""
        self.memory.add(state, action, reward, next_state, done)

        # Learn the optimal policy if samples are enough
        if len(self.memory) > BATCH_SIZE:
            experience = self.memory.sample()
            self.learn(experience, GAMMA)

    def act(self, state, add_noise = True):
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """ Update policy and value parameters using given batch of experience tuples.
        Q_target = r + gamma * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state)-> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences(Tuple[touch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        state, actions, rewards, next_state, dones = experiences

        # -------------update critic-----------------#
        # Get predicted next-state actions and Q values from target models
        action_next = self.actor_target(next_state)
        Q_target_next = self.critic_target(next_state, action_next)
        # Compute Q targets for current state (y_i)
        Q_targets = rewards + (gamma * Q_target_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(state, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ------------update actor---------------------#
        # Compute actor loss
        action_pred = self.actor_local(state)
        actor_loss = -self.critic_local(state, action_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ------------update target network------------#
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, t):
        for target_params, local_params in zip(target_model.parameters(), local_model.parameters()):
            target_params.data.copy_(t * local_params.data + (1.0 - t) * target_params.data)

class OUNoise:
    def __init__(self, size, seed, mu = 0, theta = 0.15, sigma = 0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen = buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names = ["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return(states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
