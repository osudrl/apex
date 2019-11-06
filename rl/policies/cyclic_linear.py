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

        # create actor network
        self.actorPolicies = nn.Linear( (num_inputs-1), action_dim*nPolicies)

        # create critic network
        self.criticPolicies = nn.Linear( (num_inputs-1), nPolicies)

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
        n_inputs = inputs.size(0)
        x = torch.cat( (inputs[:,:self.phase_idx],inputs[:,(self.phase_idx+1):]),1 )
        phase = inputs[:,self.phase_idx]
        a_all = self.actorPolicies(x)
        v_all = self.criticPolicies(x)

        N = torch.zeros(n_inputs, self.nPolicies*self.action_dim)
        W = torch.cat( [torch.eye(self.action_dim)]*self.nPolicies )

        Nc = torch.zeros(n_inputs, self.nPolicies)
        Wc = torch.ones(self.nPolicies,1)

        pi_1 = torch.fmod(torch.floor(phase*self.nPolicies),self.nPolicies)
        pi_2 = torch.fmod(torch.ceil(phase*self.nPolicies),self.nPolicies)
        w_1 = pi_2 - torch.fmod(phase*self.nPolicies,self.nPolicies)
        w_2 = torch.fmod(phase*self.nPolicies,self.nPolicies) - pi_1

        for i in range(n_inputs):
            N[i, (pi_1[i].int()*self.action_dim):((pi_1[i].int()+1)*self.action_dim) ] = w_1[i]
            N[i, (pi_2[i].int()*self.action_dim):((pi_2[i].int()+1)*self.action_dim) ] = w_2[i]
            Nc[i, pi_1[i].int()] = w_1[i]
            Nc[i, pi_2[i].int()] = w_2[i]

        # print("Phase, pi1, pi2")
        # print(phase)
        # print(pi_1.shape)
        # print(pi_2.shape)
        # print("w_1; w_2")
        # print(w_1.shape)
        # print(w_2.shape)
        # print("a_all")
        # print(a_all.shape)
        # print(a_all)
        # print("N")
        # print(N.shape)
        # print(N)
        # print("W")
        # print(W.shape)
        # print(W)

        temp = torch.mul(N, a_all)
        mean = torch.mm( temp, W)



        # print("v_all")
        # print(v_all.shape)
        # print(v_all)
        # print('Nc')
        # print(Nc.shape)
        # print(Nc)
        # print('Wc')
        # print(Wc.shape)
        # print(Wc)

        temp2 = torch.mul(Nc, v_all)
        # print('temp2')
        # print(temp2.shape)
        # print(temp2)
        value = torch.mm(temp2, Wc)



        return value, mean