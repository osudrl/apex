"""This screws up visualize.py"""
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from torch.autograd import Variable as Var
from torch import Tensor


class RealtimePlot():
    def __init__(self, style='ggplot'):
        plt.style.use(style)
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.xlim = 0
        self.yvals = []
        self.line = Line2D([], [])
        self.ax.add_line(self.line)

    def config(self, ylabel, xlabel):
        self.ax.set_ylabel(ylabel)
        self.ax.set_xlabel(xlabel)
        self.fig.tight_layout()

    def plot(self, y):
        self.yvals.append(y)
        self.line.set_data(np.arange(len(self.yvals)), self.yvals)

        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.set_xlim(0, self.xlim)
        self.xlim += 1

        self.fig.canvas.flush_events()

    def done(self):
        plt.ioff()
        plt.show()


def policyplot(env, policy, trj_len):
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    y = np.zeros((trj_len, action_dim))
    X = np.zeros((trj_len, obs_dim))

    obs = env.reset()
    for t in range(trj_len):
        X[t, :] = obs
        action = policy(Var(Tensor(obs[None, :]))).data.numpy()[0]
        y[t, :] = action
        obs = env.step(action)[0]

    fig, axes = plt.subplots(1, action_dim)

    for a in range(action_dim):
        axes[a].plot(np.arange(trj_len), y[:, a])

    plt.show()
"""