import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


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
