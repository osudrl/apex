import torch
from torch.autograd import Variable
import time

@torch.no_grad()
def renderpolicy(env, policy, deterministic=False, speedup=1, dt=0.05):
    state = torch.Tensor(env.reset())
    while True:
        _, action = policy.act(state, deterministic)

        state, reward, done, _ = env.step(action.data.numpy())

        if done:
            state = env.reset()

        state = torch.Tensor(state)

        env.render()

        time.sleep(dt / speedup)

def renderloop(env, policy, deterministic=False, speedup=1):
    while True:
        renderpolicy(env, policy, deterministic, speedup)