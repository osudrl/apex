import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

#from rl.utils.remote_replay import ReplayBuffer
#from rl.policies.td3_actor_critic import Original_Actor as O_Actor, TD3Critic as Critic

class DDPG():
  def __init__(self, state_dim, action_dim, max_action, a_lr, c_lr, discount=0.99, tau=0.01):
    self.behavioral_actor  = None #TODO
    self.behavioral_critic = None #TODO

    self.target_actor  = None #TODO
    self.target_critic = None #TODO

    self.actor_optimizer = torch.optim.SGD(self.behavioral_actor.parameters(), lr=a_lr)
    self.critic_optimizer = torch.optim.SGD(self.behavioral_critic.parameters(), lr=a_lr)

    self.max_action = max_action

    self.discount = discount
    self.tau = tau

  def train(self, replay_buffer, batch_size=64):
    pass

