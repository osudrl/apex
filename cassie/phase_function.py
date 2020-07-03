import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt

def create_phase_reward(swing_duration, stance_duration, strict_relaxer, stance_mode, have_incentive, FREQ=40, for_viz=False):

    total_duration = 2 * swing_duration + 2 * stance_duration

    # NOTE: these times are being converted from time in seconds to phaselength
    right_swing = np.array([0.0, swing_duration]) * FREQ
    first_dblstance = np.array([swing_duration, swing_duration + stance_duration]) * FREQ
    left_swing = np.array([swing_duration + stance_duration, 2 * swing_duration + stance_duration]) * FREQ
    second_dblstance = np.array([2 * swing_duration + stance_duration, total_duration]) * FREQ

    l_phase_points = np.zeros((2, 8))
    r_phase_points = np.zeros((2, 8))
    
    right_swing_relax_offset = (right_swing[1] - right_swing[0]) * strict_relaxer
    # l_phase_points[0,0] = right_swing[0] + right_swing_relax_offset
    # l_phase_points[0,1] = right_swing[1] - right_swing_relax_offset
    # r_phase_points[0,0] = right_swing[0] + right_swing_relax_offset
    # r_phase_points[0,1] = right_swing[1] - right_swing_relax_offset
    l_phase_points[0,0] = right_swing[0] + right_swing_relax_offset
    l_phase_points[0,1] = right_swing[1] - right_swing_relax_offset
    r_phase_points[0,0] = right_swing[0] + right_swing_relax_offset
    r_phase_points[0,1] = right_swing[1] - right_swing_relax_offset
    l_phase_points[1,:2] = np.ones(2)
    if not have_incentive:
        r_phase_points[1,:2] = np.array([-1e-5, -1e-5])
    else:
        r_phase_points[1,:2] = np.negative(np.ones(2))

    dbl_stance_relax_offset = (first_dblstance[1] - first_dblstance[0]) * strict_relaxer
    # l_phase_points[0,2] = first_dblstance[0] + dbl_stance_relax_offset
    # l_phase_points[0,3] = first_dblstance[1] - dbl_stance_relax_offset
    # r_phase_points[0,2] = first_dblstance[0] + dbl_stance_relax_offset
    # r_phase_points[0,3] = first_dblstance[1] - dbl_stance_relax_offset
    l_phase_points[0,2] = first_dblstance[0] + dbl_stance_relax_offset
    l_phase_points[0,3] = first_dblstance[1] - dbl_stance_relax_offset
    r_phase_points[0,2] = first_dblstance[0] + dbl_stance_relax_offset
    r_phase_points[0,3] = first_dblstance[1] - dbl_stance_relax_offset
    if stance_mode == "aerial":
        l_phase_points[1,2:4] = np.negative(np.ones(2))
        r_phase_points[1,2:4] = np.negative(np.ones(2))
    elif stance_mode == "zero":
        l_phase_points[1,2:4] = np.zeros(2)
        r_phase_points[1,2:4] = np.zeros(2)
    else:
        l_phase_points[1,2:4] = np.ones(2)
        r_phase_points[1,2:4] = np.ones(2)

    left_swing_relax_offset = (left_swing[1] - left_swing[0]) * strict_relaxer
    # l_phase_points[0,4] = left_swing[0] + left_swing_relax_offset
    # l_phase_points[0,5] = left_swing[1] - left_swing_relax_offset
    # r_phase_points[0,4] = left_swing[0] + left_swing_relax_offset
    # r_phase_points[0,5] = left_swing[1] - left_swing_relax_offset
    l_phase_points[0,4] = left_swing[0] + left_swing_relax_offset
    l_phase_points[0,5] = left_swing[1] - left_swing_relax_offset
    r_phase_points[0,4] = left_swing[0] + left_swing_relax_offset
    r_phase_points[0,5] = left_swing[1] - left_swing_relax_offset
    if not have_incentive:
        l_phase_points[1,4:6] = np.array([-1e-5, -1e-5])
    else:
        l_phase_points[1,4:6] = np.negative(np.ones(2))
    r_phase_points[1,4:6] = np.ones(2)

    dbl_stance_relax_offset = (second_dblstance[1] - second_dblstance[0]) * strict_relaxer
    # l_phase_points[0,6] = second_dblstance[0] + dbl_stance_relax_offset
    # l_phase_points[0,7] = second_dblstance[1] - dbl_stance_relax_offset
    # r_phase_points[0,6] = second_dblstance[0] + dbl_stance_relax_offset
    # r_phase_points[0,7] = second_dblstance[1] - dbl_stance_relax_offset
    l_phase_points[0,6] = second_dblstance[0] + dbl_stance_relax_offset
    l_phase_points[0,7] = second_dblstance[1] - dbl_stance_relax_offset
    r_phase_points[0,6] = second_dblstance[0] + dbl_stance_relax_offset
    r_phase_points[0,7] = second_dblstance[1] - dbl_stance_relax_offset
    if stance_mode == "aerial":
        l_phase_points[1,6:] = np.negative(np.ones(2))
        r_phase_points[1,6:] = np.negative(np.ones(2))
    elif stance_mode == "zero":
        l_phase_points[1,6:] = np.zeros(2)
        r_phase_points[1,6:] = np.zeros(2)
    else:          
        l_phase_points[1,6:] = np.ones(2)
        r_phase_points[1,6:] = np.ones(2)

    ## extend the data to three cycles : one before and one after : this ensures continuity

    l_prev_cycle = np.copy(l_phase_points)
    r_prev_cycle = np.copy(r_phase_points)
    l_prev_cycle[0] = l_phase_points[0] - l_phase_points[0, -1] - dbl_stance_relax_offset
    r_prev_cycle[0] = r_phase_points[0] - r_phase_points[0, -1] - dbl_stance_relax_offset

    l_second_cycle = np.copy(l_phase_points)
    r_second_cycle = np.copy(r_phase_points)
    l_second_cycle[0] = l_phase_points[0] + l_phase_points[0, -1] + dbl_stance_relax_offset
    r_second_cycle[0] = r_phase_points[0] + r_phase_points[0, -1] + dbl_stance_relax_offset
    # if mode == "strict":
    #     l_second_cycle = l_second_cycle[:, 1:]
    #     r_second_cycle = r_second_cycle[:, 1:]
    l_phase_points_repeated = np.hstack((l_prev_cycle, l_phase_points, l_second_cycle))
    r_phase_points_repeated = np.hstack((r_prev_cycle, r_phase_points, r_second_cycle))
    # repeat_time = np.linspace(0, time[-1]*2, num=time.shape[0]*2)
    
    # 2 kHz sampling for repeat time
    if for_viz:
        repeat_time = np.linspace(l_phase_points_repeated[0,0], l_phase_points_repeated[0,-1], num=int(2000*total_duration))

    ## Create the smoothing function with cubic spline and cutoff at limits -1 and 1
    l_phase_spline = PchipInterpolator(l_phase_points_repeated[0], l_phase_points_repeated[1])
    r_phase_spline = PchipInterpolator(r_phase_points_repeated[0], r_phase_points_repeated[1])

    if for_viz:
        l_phase_spline_out = np.vstack([repeat_time, l_phase_spline(repeat_time)])
        r_phase_spline_out = np.vstack([repeat_time, r_phase_spline(repeat_time)])

    ## Calculate the phaselength of this new function
    phaselength = total_duration * FREQ

    if for_viz:
        return l_phase_spline, r_phase_spline, l_phase_spline_out, r_phase_spline_out, l_phase_points_repeated, r_phase_points_repeated, repeat_time
    
    return l_phase_spline, r_phase_spline, phaselength

def encode_stance_mode(mode):
    if mode == "grounded":
        return np.array([1.,0.,0.])
    elif mode == "aerial":
        return np.array([0.,1.,0.])
    elif mode == "zero":
        return np.array([0.,0.,1.])
    else:
        print("oops")
        exit()

def decode_stance_mode(code):
    if (code == np.array([1.,0.,0.])).all():
        return "grounded"
    elif (code == np.array([0.,1.,0.])).all():
        return "aerial"
    elif (code == np.array([0.,0.,1.])).all():
        return "zero"
    else:
        print("oops2")
        exit()


def time_to_phase(x, FREQ=40):
    return x * FREQ
def phase_to_time(x, FREQ=40):
    return x / FREQ

class LivePlot:
    def __init__(self, wrap=False):

        self.wrap_viz = wrap

    def __call__(self, pipe):
        print('starting plotter...')

        self.pipe = pipe
        self.fig, self.axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10,5), constrained_layout=True)
        plt.subplots_adjust(left=0.25, bottom=0.3)
        self.axs[0].set_ylabel("left")
        self.axs[1].set_ylabel("right")
        self.second_x_axis = self.axs[0].secondary_xaxis('top', functions=(phase_to_time, time_to_phase))
        self._draw(*create_phase_reward(0.25, 0.15, 0.1, "zero", True, for_viz=True))
        timer = self.fig.canvas.new_timer(interval=1000)
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
                self._redraw(*create_phase_reward(*command, for_viz=True))
        self.fig.canvas.draw()
        return True

    def _draw(self, l_phase_spline, r_phase_spline, l_phase_spline_out, r_phase_spline_out, l_phase_points_repeated, r_phase_points_repeated, repeat_time):
        self.s0 = self.axs[0].scatter(l_phase_points_repeated[0], l_phase_points_repeated[1], color='black')
        self.s1 = self.axs[1].scatter(r_phase_points_repeated[0], r_phase_points_repeated[1], color='black')
        self.l0, = self.axs[0].plot(l_phase_spline_out[0], l_phase_spline_out[1], color='black')
        self.l1, = self.axs[1].plot(r_phase_spline_out[0], r_phase_spline_out[1], color='black')
        self._set_axis(l_phase_points_repeated)

    def _set_axis(self, points):
        if self.wrap_viz:
            self.axs[0].set_xlim(points[0, 0], points[0, -1])
            self.axs[1].set_xlim(points[0, 0], points[0, -1])
        else:
            self.axs[0].set_xlim(points[0, 7], points[0, 15])
            self.axs[1].set_xlim(points[0, 7], points[0, 15])

    def _redraw(self, l_phase_spline, r_phase_spline, l_phase_spline_out, r_phase_spline_out, l_phase_points_repeated, r_phase_points_repeated, repeat_time):
        self.l0.set_xdata(l_phase_spline_out[0])
        self.l0.set_ydata(l_phase_spline_out[1])
        self.l1.set_xdata(r_phase_spline_out[0])
        self.l1.set_ydata(r_phase_spline_out[1])
        self.s0.set_offsets(l_phase_points_repeated.T)
        self.s1.set_offsets(r_phase_points_repeated.T)
        self._set_axis(l_phase_points_repeated)
