import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Cricit

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
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 1e-3
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
WEIGHT_DECAY = 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
     
