import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time

SAMPLE_FREQ = 2000

try:
    from .periodic_func import Phase, probabilistic_periodic_func, shift_behavior
except ImportError:
    from periodic_func import Phase, probabilistic_periodic_func, shift_behavior

class InteractivePlot:
    def __init__(self):

        self.fig, self.axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10,5), constrained_layout=True)
        plt.subplots_adjust(bottom=0.4)
        self.axs[0].set_ylabel("l phase function")
        self.axs[1].set_ylabel("r phase function")

        # DIALS
        self.p1_ratio = 0.60
        self.p2_ratio = 0.40
        self.p1_coeff = 1.0
        self.p2_coeff = -1.0
        self.std = 0.01
        self.c1_shift = 0.0
        self.c2_shift = 0.5

        # axes for dials
        p1_ratio_ax     = plt.axes([0.25, 0.00, 0.65, 0.03])
        p2_ratio_ax     = plt.axes([0.25, 0.05, 0.65, 0.03])
        p1_coeff_ax     = plt.axes([0.25, 0.10, 0.65, 0.03])
        p2_coeff_ax     = plt.axes([0.25, 0.15, 0.65, 0.03])
        std_ax          = plt.axes([0.25, 0.20, 0.65, 0.03])
        c1_shift_ax     = plt.axes([0.25, 0.25, 0.65, 0.03])
        c2_shift_ax     = plt.axes([0.25, 0.30, 0.65, 0.03])

        # attach sliders to axes
        self.p1_ratio_s    = Slider(p1_ratio_ax, 'ratio', 0, 1, valinit=self.p1_ratio)
        self.p2_ratio_s    = Slider(p2_ratio_ax, 'p2_ratio', 0, 1, valinit=self.p2_ratio)
        self.p1_coeff_s    = Slider(p1_coeff_ax, 'p1_coeff', -1, 1, valinit=self.p1_coeff)
        self.p2_coeff_s    = Slider(p2_coeff_ax, 'p2_coeff', -1, 1, valinit=self.p2_coeff)
        self.std_s         = Slider(std_ax, 'std', 1e-6, 0.5, valinit=self.std)
        self.c1_shift_s    = Slider(c1_shift_ax, 'c1_shift', -0.5, 0.5, valinit=self.c1_shift)
        self.c2_shift_s    = Slider(c2_shift_ax, 'c2_shift', -0.5, 0.5, valinit=self.c2_shift)

        # attach call-back functions
        self.p1_ratio_s.on_changed(self._update_all)
        self.p2_ratio_s.on_changed(self._update_all)
        self.p1_coeff_s.on_changed(self._update_all)
        self.p2_coeff_s.on_changed(self._update_all)
        self.std_s.on_changed(self._update_all)
        self.c1_shift_s.on_changed(self._update_all)
        self.c2_shift_s.on_changed(self._update_all)

        # Domain
        self.xlim = self.p1_ratio+self.p2_ratio
        print(self.xlim)
        start = time.time()
        self.xs = np.linspace(0, self.xlim, num=SAMPLE_FREQ)
    
        # First Draw
        gait = [
            Phase(start=0.0, end=self.p1_ratio, std=self.std, coeff=self.p1_coeff),
            Phase(start=self.p1_ratio, end=self.p1_ratio+self.p2_ratio, std=self.std, coeff=self.p2_coeff)
        ]
        left = probabilistic_periodic_func(self.xs, shift_behavior(gait, shift=self.c1_shift, xlim=self.xlim))
        right = probabilistic_periodic_func(self.xs, shift_behavior(gait, shift=self.c2_shift, xlim=self.xlim))
        print(time.time()-start)
        self.l0, = self.axs[0].plot(self.xs, left)
        self.l1, = self.axs[1].plot(self.xs, right)

    def _update_all(self, _):
        
        self.p1_ratio = self.p1_ratio_s.val
        self.p2_ratio = self.p2_ratio_s.val
        # self.p2_ratio = self.xlim - self.p1_ratio
        self.p1_coeff = self.p1_coeff_s.val
        self.p2_coeff = self.p2_coeff_s.val
        self.std = self.std_s.val
        self.c1_shift = self.c1_shift_s.val
        self.c2_shift = self.c2_shift_s.val

        start = time.time()
        self.xs = np.linspace(0, self.xlim, num=SAMPLE_FREQ)
        gait = [
            Phase(start=0.0, end=self.p1_ratio, std=self.std, coeff=self.p1_coeff),
            Phase(start=self.p1_ratio, end=self.p1_ratio+self.p2_ratio, std=self.std, coeff=self.p2_coeff)
        ]
        self.xlim = self.p1_ratio + self.p2_ratio
        left = probabilistic_periodic_func(self.xs, shift_behavior(gait, shift=self.c1_shift, xlim=self.xlim))
        right = probabilistic_periodic_func(self.xs, shift_behavior(gait, shift=self.c2_shift, xlim=self.xlim))
        print(time.time()-start)
        self.axs[0].clear()
        self.axs[1].clear()
        self.l0, = self.axs[0].plot(self.xs, left)
        self.l1, = self.axs[1].plot(self.xs, right)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider  # , Button, RadioButtons
    plot = InteractivePlot()
    plt.show()

    S = 1   # swing
    G = -1  # stance
    hopping = [
        Phase(start=0.0, end=0.5, sigma=0.01, coeff=G),
        Phase(start=0.5, end=1.0, sigma=0.01, coeff=S)
    ]

    walking = [
        Phase(start=0.0, end=0.4, sigma=0.01, coeff=S),
        Phase(start=0.4, end=1.0, sigma=0.01, coeff=G)
    ]

    running = [
        Phase(start=0.0, end=0.7, sigma=0.01, coeff=1),
        Phase(start=0.7, end=1.0, sigma=0.01, coeff=-1)
    ]

    skipping = [
        Phase(start=0.0, end=0.2, sigma=0.01, coeff=G),
        Phase(start=0.2, end=0.6, sigma=0.01, coeff=S),
        Phase(start=0.6, end=0.8, sigma=0.01, coeff=G),
        Phase(start=0.8, end=1.0, sigma=0.01, coeff=S),
    ]

    x = np.linspace(0, 1, num=2000)
    fig, axs = plt.subplots(nrows=4, ncols=1, sharex=True)
    start_prob = norm.cdf(x, loc=0.0, scale=0.01)  # prob that we are past start of phase
    end_prob = norm.cdf(x, loc=0.7, scale=0.01)  # prob that we are past end of phase
    axs[0].plot(x, 1 * (start_prob - end_prob))
    start_prob = norm.cdf(x, loc=0.7, scale=0.01)  # prob that we are past start of phase
    end_prob = norm.cdf(x, loc=1.0, scale=0.01)  # prob that we are past end of phase
    axs[1].plot(x, (
        -1 * (start_prob - end_prob)
    ))
    axs[2].plot(x, probabilistic_periodic_func(x, running))
    axs[3].plot(x, probabilistic_periodic_func(x, shift_behavior(running, shift=0.5)))
    plt.show()
