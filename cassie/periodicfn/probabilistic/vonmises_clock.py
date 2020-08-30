import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

SAMPLE_FREQ = 2000

import time

try:
    from .periodic_func import Phase, vonmises_func
except ImportError:
    from periodic_func import Phase, vonmises_func

class InteractivePlot:
    def __init__(self):

        self.fig, self.axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10,5), constrained_layout=True)
        plt.subplots_adjust(bottom=0.4)
        self.axs[0].set_ylabel("l phase function")
        self.axs[1].set_ylabel("r phase function")

        # DIALS
        self.p1_ratio = 0.50
        self.p2_ratio = 0.50
        self.p1_coeff = 1.0
        self.p2_coeff = -1.0
        self.std = 0.1
        self.c1_shift = 0.0
        self.c2_shift = 0.5

        # axes for dials
        p1_ratio_ax     = plt.axes([0.25, 0.05, 0.65, 0.03])
        p2_ratio_ax     = plt.axes([0.25, 0.10, 0.65, 0.03])
        p1_coeff_ax     = plt.axes([0.25, 0.15, 0.65, 0.03])
        p2_coeff_ax     = plt.axes([0.25, 0.20, 0.65, 0.03])
        std_ax          = plt.axes([0.25, 0.25, 0.65, 0.03])
        c1_shift_ax     = plt.axes([0.25, 0.30, 0.65, 0.03])
        c2_shift_ax     = plt.axes([0.25, 0.35, 0.65, 0.03])

        # attach sliders to axes
        self.p1_ratio_s    = Slider(p1_ratio_ax,    'p1_ratio',          0, 1,   valinit=self.p1_ratio)
        self.p2_ratio_s    = Slider(p2_ratio_ax,    'p2_ratio',          0, 1,   valinit=self.p2_ratio)
        self.p1_coeff_s    = Slider(p1_coeff_ax,    'p1_coeff',      -1, 1,   valinit=self.p1_coeff)
        self.p2_coeff_s    = Slider(p2_coeff_ax,    'p2_coeff',      -1, 1,   valinit=self.p2_coeff)
        self.std_s         = Slider(std_ax,       'variance',       1e-3, 0.5, valinit=self.std)
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
        self.xs = np.linspace(0, self.xlim, num=SAMPLE_FREQ)
    
        # First Draw
        gait = [
            Phase(start=0.0,           end=self.p1_ratio,                 std=self.std, coeff=self.p1_coeff),
            Phase(start=self.p1_ratio, end=self.p1_ratio + self.p2_ratio, std=self.std, coeff=self.p2_coeff)
        ]

        left = vonmises_func(self.xs, gait, shift=self.c1_shift)
        right = vonmises_func(self.xs, gait, shift=self.c2_shift)

        self.l0, = self.axs[0].plot(self.xs, left)
        self.l1, = self.axs[1].plot(self.xs, right)

    def _update_all(self, _):
        
        self.p1_ratio    = self.p1_ratio_s.val
        self.p2_ratio    = self.p2_ratio_s.val
        self.p1_coeff    = self.p1_coeff_s.val
        self.p2_coeff    = self.p2_coeff_s.val
        self.std = self.std_s.val
        self.c1_shift = self.c1_shift_s.val
        self.c2_shift = self.c2_shift_s.val

        start = time.time()

        self.xs = np.linspace(0, self.xlim, num=SAMPLE_FREQ)
        gait = [
            Phase(start=0.0,           end=self.p1_ratio,                 std=self.std, coeff=self.p1_coeff),
            Phase(start=self.p1_ratio, end=self.p1_ratio + self.p2_ratio, std=self.std, coeff=self.p2_coeff)
        ]

        left = vonmises_func(self.xs, gait, shift=self.c1_shift)
        right = vonmises_func(self.xs, gait, shift=self.c2_shift)

        print(time.time() - start)

        self.axs[0].clear()
        self.axs[1].clear()
        self.l0, = self.axs[0].plot(self.xs, left)
        self.l1, = self.axs[1].plot(self.xs, right)

class LivePlot:
    def __init__(self, wrap=False):

        self.wrap_viz = wrap

    def __call__(self, pipe):
        print('starting plotter...')

        self.pipe = pipe
        self.fig, self.axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10,5))
        self.axs[0].set_title('Live PeriodicFcn Plot')
        self.axs[0].set_ylabel("left")
        self.axs[1].set_ylabel("right")
        
        # domain
        self.ratios = [0.5, 0.5]
        self.x = np.linspace(0, sum(self.ratios), num=SAMPLE_FREQ)

        # first draw
        gait = [
            Phase(start=0.0, end=0.5, std=0.01, coeff=-1),
            Phase(start=0.5, end=1.0, std=0.01, coeff=1)
        ]
        left_foot = vonmises_func(self.x, gait, shift=0.0)
        right_foot = vonmises_func(self.x, gait, shift=0.5)
        self._draw(left_foot, right_foot)

        timer = self.fig.canvas.new_timer(interval=10)
        timer.add_callback(self.call_back)
        timer.start()

        print('...done')
        plt.show()

    def terminate(self):
        plt.close('all')

    def call_back(self):
        while self.pipe.poll():
            command = self.pipe.recv()
            if command is None:
                self.terminate()
                return False
            else:
                # coeffs, self.ratios, ratio_shift, stds = command
                # gait = [
                #     Phase(start=0.0, end=self.ratios[0], std=stds[0], coeff=coeffs[0]),
                #     Phase(start=self.ratios[0], end=self.ratios[0]+self.ratios[1], std=stds[1], coeff=coeffs[1]),
                # ]
                gait, ratio_shift = command
                left_foot = vonmises_func(self.x, gait, shift=ratio_shift[0])
                right_foot = vonmises_func(self.x, gait, shift=ratio_shift[1])
                self._redraw(left_foot, right_foot)
        self.fig.canvas.draw()
        return True

    def _draw(self, left_foot, right_foot):
        self.x = np.linspace(0, sum(self.ratios), num=SAMPLE_FREQ)
        self.l0, = self.axs[0].plot(self.x, left_foot)
        self.l1, = self.axs[1].plot(self.x, right_foot)

    def _redraw(self, left_foot, right_foot):
        # get right foot force / vel data
        self.x = np.linspace(0, sum(self.ratios), num=SAMPLE_FREQ)
        self.l0.set_xdata(self.x)
        self.l1.set_xdata(self.x)
        self.l0.set_ydata(left_foot)
        self.l1.set_ydata(right_foot)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider  # , Button, RadioButtons
    plot = InteractivePlot()
    plt.show()
    exit()
