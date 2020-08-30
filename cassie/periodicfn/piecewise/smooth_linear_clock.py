import numpy as np
import time

SAMPLE_FREQ = 2000

# There probably already is a decorator for this somewhere in numpy... seems like a no-brainer
def vectorized_func(func):
    vfunc = np.vectorize(func, doc=f'Vectorized `{func.__name__}`')
    return vfunc

# TODO: Replace flip with some sort of phase-offset

@vectorized_func
def linear_piecewise_clock(phi, ratio=0.5, alpha=0.05, flip=False):
    # domain for period is x: [0,1]
    beta = 0.0
    if flip:
        beta = 0.5
    phi = np.fmod(phi + beta, 1)

    saturation = alpha * (ratio / 2 - 1e-3)
    slope = 1 / ((ratio / 2) - saturation)

    if phi < saturation:
        return 1.0
    elif phi < ratio/2:
        return 1 - slope * (phi - saturation)
    elif phi < 1 - ratio/2:
        return 0.0
    elif phi < 1 - saturation:
        return 1 + slope * (phi + saturation - 1)
    else:
        return 1.0

@vectorized_func
def piecewise_sin(x, ratio=0.7, flip=False):
    beta = 0.0 if not flip else 0.5
    x = np.fmod(x+beta,1)

    if x < ratio:
        return np.sin(np.pi * 1/ratio * x)
    elif x > ratio:
        return -np.sin(np.pi * 1/(1-ratio) * (x-ratio))
    else:
        return 0

def smooth_square_wave(phi, alpha=0.05, ratio=0.5, flip=False):

    clock = piecewise_sin(phi, ratio=ratio, flip=flip)
    return (1 / np.arctan(1 / alpha)) * np.arctan(clock / alpha)

class InteractivePlot:
    def __init__(self):

        self.fig, self.axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10,5), constrained_layout=True)
        plt.subplots_adjust(bottom=0.4)
        self.axs[0].set_ylabel("l phase function")
        self.axs[1].set_ylabel("r phase function")
        # self.axs[2].set_ylabel("aerial and grounded")
        # self.axs[3].set_ylabel("single swing")

        # DIALS
        self.period_num = 1         # number of periods to plot
        self.period_time = 1.0      # how many seconds a single period lasts for
        self.ratio = 0.60           # ratio between phases
        self.alpha = 0.01           # saturation parameter
        # axes for dials
        period_num_ax = plt.axes([0.25, 0.05, 0.65, 0.03])
        period_time_ax = plt.axes([0.25, 0.10, 0.65, 0.03])
        ratio_ax = plt.axes([0.25, 0.15, 0.65, 0.03])
        alpha_ax = plt.axes([0.25, 0.20, 0.65, 0.03])
        # attach sliders to axes
        self.period_num_s = Slider(period_num_ax, 'period num', 0, 4, valinit=self.period_num, valstep=1)
        self.period_time_s = Slider(period_time_ax, 'period time', 0.1, 1.8, valinit=self.period_time)
        self.ratio_s = Slider(ratio_ax, 'ratio', 0, 1, valinit=self.ratio)
        self.alpha_s = Slider(alpha_ax, 'alpha', 0, 1-1e-5, valinit=self.alpha)
        # attach call-back functions
        self.period_num_s.on_changed(self._update_all)
        self.period_time_s.on_changed(self._update_all)
        self.ratio_s.on_changed(self._update_range)
        self.alpha_s.on_changed(self._update_range)

        # Domain
        self.xs = np.linspace(0, self.period_time*self.period_num, num=SAMPLE_FREQ)
    
        # First Draw
        start = time.time()
        l_phase = smooth_square_wave(self.xs, ratio=self.ratio, alpha=self.alpha, flip=False)
        r_phase = smooth_square_wave(self.xs, ratio=self.ratio, alpha=self.alpha, flip=True)
        print(time.time()-start)
        self.l0, = self.axs[0].plot(self.xs * self.period_time, l_phase)
        self.l1, = self.axs[1].plot(self.xs * self.period_time, r_phase)

    def _update_range(self, _):
        self.ratio = self.ratio_s.val
        self.alpha = self.alpha_s.val
        start = time.time()
        l_phase = smooth_square_wave(self.xs, ratio=self.ratio, alpha=self.alpha, flip=False)
        r_phase = smooth_square_wave(self.xs, ratio=self.ratio, alpha=self.alpha, flip=True)
        print(time.time()-start)
        self.l0.set_ydata(l_phase)
        self.l1.set_ydata(r_phase)

    def _update_all(self, _):
        self.period_num = self.period_num_s.val
        self.period_time = self.period_time_s.val
        self.xs = np.linspace(0, self.period_num, num=SAMPLE_FREQ)
        l_phase = smooth_square_wave(self.xs, ratio=self.ratio, alpha=self.alpha, flip=False)
        r_phase = smooth_square_wave(self.xs, ratio=self.ratio, alpha=self.alpha, flip=True)
        self.l0.set_xdata(self.xs * self.period_time)
        self.l0.set_ydata(l_phase)
        self.l1.set_xdata(self.xs * self.period_time)
        self.l1.set_ydata(r_phase)
        _ = [ax.set_xlim(0, self.period_time*self.period_num) for ax in self.axs]

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  from matplotlib.widgets import Slider  # , Button, RadioButtons
  plot = InteractivePlot()
  plt.show()

  """

  period_time = 1.0    # how many seconds a single period lasts for
  phase_num = 4        # how many phases in a period
  period_num = 2       # how many periods to plot (just 1 makes repeats hard to see)
  sample_freq = 2000

  ratio = 0.8
  alpha = 0.99

  xs = np.linspace(0, period_time*period_num, num=period_num*phase_num*sample_freq)
  clock = piecewise_sin(xs)
  plt.plot(xs, ys)
  ys = linear_piecewise_clock(xs, ratio=ratio, alpha=alpha)

  plt.plot(xs * period_time, ys)
  plt.plot(xs * period_time, clock)
  fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10,5), constrained_layout=True)
  axs[0].plot(xs * period_time, smooth_square_wave(xs, ratio=ratio, alpha=alpha, flip=False))
  axs[1].plot(xs * period_time, smooth_square_wave(xs, ratio=ratio, alpha=alpha, flip=True))
  plt.show()

  """
