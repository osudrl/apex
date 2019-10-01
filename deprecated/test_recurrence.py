import torch
import torch.nn as nn
import numpy as np
from rl.policies.recurrent import RecurrentNet

x = torch.ones(1, 4)
actor = RecurrentNet(4, 3)

actions = []
loss = nn.MSELoss()

for i in range(5):
  actions.append(actor.forward(x))
  print(actions)

actions = torch.cat([action for action in actions])
print(actions)
target = torch.randn(5, 3)

grad = loss(actions, target)
grad.backward()
print("success!")
