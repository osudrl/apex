import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.distributions import DiagonalGaussian
from .base import FFPolicy

import time

import numpy as np

# NOTE: the fact that this has the same name as a parameter caused a NASTY bug
# apparently "if <function_name>" evaluates to True in python...
def normc_fn(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class CyclicLinear(FFPolicy):
    def __init__(self, 
                 num_inputs, 
                 action_dim,
                 phase_idx, 
                 init_std=1,
                 nPolicies=10, 
                 learn_std=True,
                 normc_init=True,
                 obs_std=None,
                 obs_mean=None):
        super(CyclicLinear, self).__init__()

        self.nPolicies = nPolicies
        self.phase_idx = phase_idx
        self.action_dim = action_dim
        self.actorPolicies = []
        self.criticPolicies = []

        # create actor network
        for idx in range(nPolicies):
            self.actorPolicies.append(nn.Linear(num_inputs-1, action_dim))

        # create critic network
        for idx in range(nPolicies):
            self.criticPolicies.append(nn.Linear(num_inputs-1, 1))

        self.dist = DiagonalGaussian(action_dim, init_std, learn_std)

        self.obs_std = obs_std
        self.obs_mean = obs_mean
        
        # weight initialization scheme used in PPO paper experiments
        self.normc_init = normc_init

        self.init_parameters()
        self.train()

    def init_parameters(self):
        if self.normc_init:
            print("Doing norm column initialization.")
            self.apply(normc_fn)

            if self.dist.__class__.__name__ == "DiagGaussian":
                self.mean.weight.data.mul_(0.01)

    def forward(self, inputs):
        # print(inputs.shape)
        x = torch.cat( (inputs[:,:self.phase_idx],inputs[:,(self.phase_idx+1):]),1 )
        # print(inputs[:,:self.phase_idx])
        # print(inputs[:,(self.phase_idx+1):])
        # print(x.shape)
        phase = inputs[:,self.phase_idx]
        pi_1 = torch.fmod(torch.floor(phase*self.nPolicies),self.nPolicies)
        pi_2 = torch.fmod(torch.ceil(phase*self.nPolicies),self.nPolicies)
        w_1 = pi_2 - phase*self.nPolicies
        w_2 = phase*self.nPolicies - pi_1

        # print("Phase, pi1, pi2")
        # print(phase)
        # print(pi_1.shape)
        # print(pi_2.shape)
        # print(w_1.shape)
        # print(w_2.shape)

        value = torch.zeros(pi_1.size(0), 1)
        mean = torch.zeros(pi_1.size(0), self.action_dim)

        if pi_1.size(0) > 1:
            for idx in range(pi_1.size(0)):
                value[idx,:] = w_1[idx]*self.criticPolicies[pi_1[idx].int()](x[idx,:]) + w_2[idx]*self.criticPolicies[pi_2[idx].int()](x[idx,:]) 
                mean[idx, :] = w_1[idx]*self.actorPolicies[pi_1[idx].int()](x[idx,:])  + w_2[idx]*self.actorPolicies[pi_2[idx].int()](x[idx,:]) 
        else:
            value = w_1*self.criticPolicies[pi_1.int()](x) + w_2*self.criticPolicies[pi_2.int()](x) 
            mean = w_1*self.actorPolicies[pi_1.int()](x)  + w_2*self.actorPolicies[pi_2.int()](x)
        # if self.training == False:
        #     inputs = (inputs - self.obs_mean) / self.obs_std

        return value, mean