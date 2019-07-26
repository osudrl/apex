# from https://github.com/noagarcia/visdom-tutorial/blob/master/utils.py

from visdom import Visdom

import numpy as np

import ray


class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name, port):
        self.viz = Visdom(port=port)
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, x_var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel=x_var_name,
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')