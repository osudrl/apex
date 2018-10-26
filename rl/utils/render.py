import torch
from torch.autograd import Variable
import time

def renderpolicy(env, policy, explore=True, speedup=1, dt=0.05):
    with torch.no_grad():
        state = torch.Tensor(env.reset())
        while True:
            _, action = policy.act(state)

            state, reward, done, _ = env.step(action.data.numpy())

            if done:
                state = env.reset()

            state = torch.Tensor(state)

            env.render()

            time.sleep(dt / speedup)

def renderloop(env, policy, explore=False, speedup=1):
    while True:
        renderpolicy(env, policy, explore, speedup)