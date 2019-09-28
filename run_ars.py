from rl.algos.ars import ARS
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
timesteps = 0

class Policy(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_size=32):
		super(Policy, self).__init__()

		self.l1 = nn.Linear(state_dim, hidden_size)
		#self.l2 = nn.Linear(hidden_size, hidden_size)
		self.l3 = nn.Linear(hidden_size, action_dim)
	
	def forward(self, state):
		a = F.relu(self.l1(state))
		#a = F.relu(self.l2(a))
		return torch.tanh(self.l3(a)) 


def eval_fn(policy, env, visualize=False, reward_shift=0):

  state = torch.tensor(env.reset()).float()
  rollout_reward = 0
  done = False

  while not done:
    action = policy.forward(state).detach()

    if visualize:
      env.render()
    
    state, reward, done, _ = env.step(action)
    state = torch.tensor(state).float()
    rollout_reward += reward - reward_shift
    global timesteps
    timesteps+=1
  return rollout_reward

env    = gym.make("Hopper-v2")
policy = Policy(env.observation_space.shape[0], env.action_space.shape[0])
policy = policy.float()

iters = 1000

def train(policy, env):
  algo = ARS(policy, env, deltas=100)

  def black_box(p):
    return eval_fn(p, env, reward_shift=1)

  for i in range(iters):
    algo.step(black_box)
    print("iter {} reward {} timesteps {}".format(i, eval_fn(algo.policy, env), timesteps))

train(policy, env)
