import numpy as np

class WrapEnv:
    def __init__(self, env_fn):
        self.env = env_fn()

    def step(self, action):
        state, reward, done, info = self.env.step(action[0])
        return np.array([state]), np.array([reward]), np.array([done]), np.array([info])

    def render(self):
        self.env.render()

    def reset(self):
        return np.array([self.env.reset()])