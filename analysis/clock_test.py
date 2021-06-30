import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import math

def clock_fn(swing_ratio, phase, phaselen):
    # Left is swing and then stance, right is in stance and then swing
    left_stance_t = swing_ratio*phaselen
    right_swing_t = 1 - swing_ratio

    smooth_ratio = 0.1     # What percentage of phaselen is used for smoothing in each direction
    s_offset = smooth_ratio * phaselen      # Offset in phaselen to account for smoothing
    steep = -3

    # Left side clock
    if phase < left_stance_t - s_offset:
        l_swing = 1
        l_stance = 0
    elif left_stance_t - s_offset < phase < left_stance_t + s_offset:
        l_stance = 1 - 1 / (1 + np.power(2*s_offset/(phase-(left_stance_t - s_offset)) - 1, steep))
        l_swing = 1 - l_stance
    elif left_stance_t + s_offset < phase < (1 - 2*smooth_ratio)*phaselen:
        l_swing = 0
        l_stance = 1
    elif (1 - 2*smooth_ratio)*phaselen < phase < phaselen:
        l_swing = 1 - 1 / (1 + np.power(2*s_offset/(phase-((1 - 2*smooth_ratio)*phaselen)) - 1, steep))
        l_stance = 1 - l_swing
    else:   # phase = phaselen, wrap back around
        l_swing = 1
        l_stance = 0

    return l_swing, l_stance

def lin_clock(swing_ratio, phase, phaselen):
    # swing_ratio = 0.5

    percent_trans = .2
    swing_time = phaselen*(1-percent_trans)*swing_ratio
    stance_time = phaselen*(1-percent_trans)*(1-swing_ratio)
    trans_time = phaselen*((percent_trans)/2)
    swing_linclock = 1
    stance_linclock = 0
    if phase < swing_time:
        swing_linclock = 1
        stance_linclock = 0
    elif swing_time <= phase < swing_time+trans_time:
        swing_linclock = 1 - (phase-swing_time) / trans_time
        stance_linclock = (phase-swing_time) / trans_time
    elif swing_time+trans_time <= phase < swing_time+trans_time+stance_time:
        swing_linclock = 0
        stance_linclock = 1
    elif swing_time+trans_time+stance_time <= phase < phaselen:
        swing_linclock = (phase-(swing_time+trans_time+stance_time)) / trans_time
        stance_linclock = 1-((phase-(swing_time+trans_time+stance_time)) / trans_time)

    return swing_linclock, stance_linclock

def lin_clock2(swing_ratio, phase, phaselen):

    percent_trans = .2
    swing_time = phaselen * swing_ratio
    stance_time = phaselen * (1-swing_ratio)
    swing_time -= phaselen*percent_trans/2
    stance_time -= phaselen*percent_trans/2
    # swing_time = phaselen*(1-percent_trans)*swing_ratio
    # stance_time = phaselen*(1-percent_trans)*(1-swing_ratio)
    trans_time = phaselen*((percent_trans)/2)
    l_swing_linclock = 1
    if phase < swing_time:
        l_swing_linclock = 1
    elif swing_time <= phase < swing_time+trans_time:
        l_swing_linclock = 1 - (phase-swing_time) / trans_time
    elif swing_time+trans_time <= phase < swing_time+trans_time+stance_time:
        l_swing_linclock = 0
    elif swing_time+trans_time+stance_time <= phase < phaselen:
        l_swing_linclock = (phase-(swing_time+trans_time+stance_time)) / trans_time
    r_swing_linclock = 0
    if phase < stance_time:
        r_swing_linclock = 0
    elif stance_time <= phase < stance_time+trans_time:
        r_swing_linclock = (phase-stance_time) / trans_time
    elif stance_time+trans_time <= phase < swing_time+trans_time+stance_time:
        r_swing_linclock = 1
    elif swing_time+trans_time+stance_time <= phase < phaselen:
        r_swing_linclock = 1 - (phase-(swing_time+trans_time+stance_time)) / trans_time

    return l_swing_linclock, r_swing_linclock

def lin_clock3(swing_ratio, phase, phaselen):

    percent_trans = .2
    swing_time = phaselen * swing_ratio
    stance_time = phaselen * (1-swing_ratio)
    swing_time -= phaselen*percent_trans/2
    stance_time -= phaselen*percent_trans/2
    # swing_time = phaselen*(1-percent_trans)*swing_ratio
    # stance_time = phaselen*(1-percent_trans)*(1-swing_ratio)
    trans_time = phaselen*((percent_trans)/2)
    phase_offset = (swing_time - stance_time) / 2
    r_phase = phase - phase_offset
    if r_phase < 0:
        r_phase += phaselen
    l_swing_linclock = 1
    if phase < swing_time:
        l_swing_linclock = 1
    elif swing_time <= phase < swing_time+trans_time:
        l_swing_linclock = 1 - (phase-swing_time) / trans_time
    elif swing_time+trans_time <= phase < swing_time+trans_time+stance_time:
        l_swing_linclock = 0
    elif swing_time+trans_time+stance_time <= phase < phaselen:
        l_swing_linclock = (phase-(swing_time+trans_time+stance_time)) / trans_time
    r_swing_linclock = 0
    if r_phase < stance_time:
        r_swing_linclock = 0
    elif stance_time <= r_phase < stance_time+trans_time:
        r_swing_linclock = (r_phase-stance_time) / trans_time
    elif stance_time+trans_time <= r_phase < swing_time+trans_time+stance_time:
        r_swing_linclock = 1
    elif swing_time+trans_time+stance_time <= r_phase < phaselen:
        r_swing_linclock = 1 - (r_phase-(swing_time+trans_time+stance_time)) / trans_time

    return l_swing_linclock, r_swing_linclock

def lin_clock4(swing_ratio, phase, phaselen):

    percent_trans = 0#.2  # percent of stance time to use as transition
    swing_time = phaselen * swing_ratio
    stance_time = phaselen * (1-swing_ratio)
    trans_time = stance_time * percent_trans
    phase_offset = (swing_time - stance_time) / 2
    stance_time -= trans_time
    r_phase = phase - phase_offset
    if r_phase < 0:
        r_phase += phaselen
    l_swing_linclock = 0
    if phase < (swing_time + trans_time)/2:
        l_swing_linclock = phase / ((swing_time+trans_time)/2)
    elif (swing_time + trans_time)/2 <= phase < swing_time + trans_time:
        l_swing_linclock = 1 - (phase-(swing_time + trans_time)/2) / ((swing_time+trans_time)/2)
    elif swing_time+trans_time <= phase < phaselen:#swing_time+trans_time+stance_time:
        l_swing_linclock = 0
    r_swing_linclock = 0
    if r_phase < stance_time:
        r_swing_linclock = 0
    elif stance_time <= r_phase < stance_time + (swing_time+trans_time)/2:
        r_swing_linclock = (r_phase-stance_time) / ((swing_time+trans_time)/2)
    elif stance_time+(swing_time+trans_time)/2 <= r_phase < phaselen:
        r_swing_linclock = 1 - (r_phase-(stance_time+(swing_time+trans_time)/2)) / ((swing_time+trans_time)/2)

    return l_swing_linclock, r_swing_linclock

def lin_clock5(swing_ratio, phase, phaselen):

    percent_trans = .2 # percent of swing time to use as transition
    swing_time = phaselen * swing_ratio
    stance_time = phaselen * (1-swing_ratio)
    trans_time = swing_time * percent_trans
    phase_offset = (swing_time - stance_time) / 2
    swing_time -= trans_time
    r_phase = phase - phase_offset
    if r_phase < 0:
        r_phase += phaselen
    l_swing_linclock = 0
    if phase < trans_time / 2:
        l_swing_linclock = phase / (trans_time / 2)
    elif trans_time / 2 < phase <= swing_time + trans_time / 2:
        l_swing_linclock = 1
    elif swing_time + trans_time / 2 < phase <= swing_time + trans_time:
        l_swing_linclock = 1 - (phase-(swing_time+trans_time/2)) / (trans_time/2)
    elif swing_time+trans_time <= phase < phaselen:#swing_time+trans_time+stance_time:
        l_swing_linclock = 0
    r_swing_linclock = 0
    if r_phase < stance_time:
        r_swing_linclock = 0
    elif stance_time < r_phase <= stance_time + (trans_time)/2:
        r_swing_linclock = (r_phase-stance_time) / (trans_time/2)
    elif stance_time+trans_time/2 < r_phase <= stance_time+trans_time/2+swing_time:
        r_swing_linclock = 1
    elif stance_time+trans_time/2+swing_time < r_phase <= stance_time+swing_time+trans_time:
        r_swing_linclock = 1 - (r_phase-(stance_time+trans_time/2+swing_time)) / (trans_time/2)

    return l_swing_linclock, r_swing_linclock

def plot_clock():

    phaselen = 30
    swing_ratio = .8
    length = 100
    phases = np.linspace(0, phaselen, length)
    swing = np.zeros(length)
    stance = np.zeros(length)
    for i in range(length):
        y, y2 = clock_fn(swing_ratio, phases[i], phaselen)
        swing[i] = y
        stance[i] = y2

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(phases, swing)
    ax[1].plot(phases, stance)

    plt.show()

def plot_smooth_clock():
    x = np.linspace(0, 1, 100)
    one2one_clock = 0.5*(np.cos(2*np.pi*x) + 1)
    zero2zero_clock = 0.5*(np.cos(2*np.pi*(x-1/2)) + 1)

    fig, ax = plt.subplots(1, 1)
    ax.plot(x, one2one_clock, label="Swing Cost")
    ax.plot(x, zero2zero_clock, label="Stance Cost")
    ax.legend(loc="upper right")
    ax.set_ylabel("Cost Weighting")
    ax.set_xlabel("Cycle Time")
    plt.show()
    # plt.savefig("./noref_clock.png")

def plot_lin_clock():
    phaselen = 30
    swing_ratio = .6
    length = 100
    phases = np.linspace(0, phaselen, length)
    swing = np.zeros(length)
    stance = np.zeros(length)
    for i in range(length):
        y, y2 = lin_clock(swing_ratio, phases[i], phaselen)
        swing[i] = y
        stance[i] = y2

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(phases, swing)
    ax[1].plot(phases, stance)

    plt.show()

def plot_lin_clock2():
    phaselen = 30
    swing_ratio = 0.8
    length = 100
    num_cycles = 2
    phases = np.linspace(0, num_cycles*phaselen, length)
    l_swing = np.zeros(length)
    l_stance = np.zeros(length)
    r_swing = np.zeros(length)
    r_stance = np.zeros(length)
    for i in range(length):
        l_swing[i], r_swing[i] = lin_clock2(swing_ratio, phases[i], phaselen)
        # r_swing[i] = lin_clock2(swing_ratio, phaselen - phases[i], phaselen)
        l_stance[i] = -l_swing[i] + 1
        r_stance[i] = -r_swing[i] + 1

    # l_stance = -l_swing + 1
    # r_swing = np.flip(l_swing)
    # r_stance = -r_swing + 1

    fig, ax = plt.subplots(2, 1)
    t = np.linspace(0, 0.841, length)
    ax[0].plot(t, l_swing, label="l_swing")
    ax[0].plot(t, l_stance, label="l_stance")
    ax[1].plot(t, r_swing, label="r_swing")
    ax[1].plot(t, r_stance, label="r_stance")

    # ax[0].scatter(t, l_swing, label="l_swing")
    # ax[0].scatter(t, l_stance, label="l_stance")
    # ax[1].scatter(t, r_swing, label="r_swing")
    # ax[1].scatter(t, r_stance, label="r_stance")
    
    print(r_stance[0], r_stance[-1])

    ax[0].legend()
    ax[1].legend()
    
    # plt.show()
    plt.savefig("./clock_half_gallop.png")

def plot_lin_clock3():
    phaselen = 30
    swing_ratio = 0.5
    length = 200
    num_cycles = 1
    cycle_time = 1 
    phases = np.linspace(0, num_cycles*phaselen, length)
    l_swing = np.zeros(length)
    l_stance = np.zeros(length)
    r_swing = np.zeros(length)
    r_stance = np.zeros(length)
    for i in range(length):
        l_swing[i], r_swing[i] = lin_clock3(swing_ratio, phases[i], phaselen)
        # r_swing[i] = lin_clock2(swing_ratio, phaselen - phases[i], phaselen)
        l_stance[i] = -l_swing[i] + 1
        r_stance[i] = -r_swing[i] + 1

    # l_stance = -l_swing + 1
    # r_swing = np.flip(l_swing)
    # r_stance = -r_swing + 1

    fig, ax = plt.subplots(2, 1)
    t = np.linspace(0, cycle_time, length)
    ax[0].plot(t, l_swing, label="l_swing")
    ax[0].plot(t, l_stance, label="l_stance")
    ax[1].plot(t, r_swing, label="r_swing")
    ax[1].plot(t, r_stance, label="r_stance")

    # ax[0].scatter(t, l_swing, label="l_swing")
    # ax[0].scatter(t, l_stance, label="l_stance")
    # ax[1].scatter(t, r_swing, label="r_swing")
    # ax[1].scatter(t, r_stance, label="r_stance")
    
    print(r_stance[0], r_stance[-1])

    ax[0].legend()
    ax[1].legend()
    
    # plt.show()
    plt.savefig("./linclock3.png")

def plot_lin_clock4():
    phaselen = 30
    swing_ratio = 0.8
    length = 200
    num_cycles = 1
    cycle_time = 1 / 2
    phases = np.linspace(0, num_cycles*phaselen, length)
    l_swing = np.zeros(length)
    l_stance = np.zeros(length)
    r_swing = np.zeros(length)
    r_stance = np.zeros(length)
    for i in range(length):
        l_swing[i], r_swing[i] = lin_clock4(swing_ratio, phases[i], phaselen)
        # r_swing[i] = lin_clock2(swing_ratio, phaselen - phases[i], phaselen)
        l_stance[i] = -l_swing[i] + 1
        r_stance[i] = -r_swing[i] + 1

    # l_stance = -l_swing + 1
    # r_swing = np.flip(l_swing)
    # r_stance = -r_swing + 1

    fig, ax = plt.subplots(2, 1)
    t = np.linspace(0, cycle_time, length)
    ax[0].plot(t, l_swing, label="l_swing")
    ax[0].plot(t, l_stance, label="l_stance")
    ax[1].plot(t, r_swing, label="r_swing")
    ax[1].plot(t, r_stance, label="r_stance")

    # ax[0].scatter(t, l_swing, label="l_swing")
    # ax[0].scatter(t, l_stance, label="l_stance")
    # ax[1].scatter(t, r_swing, label="r_swing")
    # ax[1].scatter(t, r_stance, label="r_stance")
    
    print(r_stance[0], r_stance[-1])

    ax[0].legend()
    ax[1].legend()
    
    plt.show()

def plot_lin_clock5():
    phaselen = 32
    swing_ratio = 0.4
    length = 32#200
    num_cycles = 1
    cycle_time = 1
    phases = np.linspace(0, num_cycles*phaselen, length)
    l_swing = np.zeros(length)
    l_stance = np.zeros(length)
    r_swing = np.zeros(length)
    r_stance = np.zeros(length)
    l_force = np.zeros(length)
    r_force = np.zeros(length)
    for i in range(length):
        l_swing[i], r_swing[i] = lin_clock5(swing_ratio, phases[i], phaselen)
        l_force[i] = math.ceil(l_swing[i])
        r_force[i] = math.ceil(r_swing[i])
        # r_swing[i] = lin_clock2(swing_ratio, phaselen - phases[i], phaselen)
        l_stance[i] = -l_swing[i] + 1
        r_stance[i] = -r_swing[i] + 1

    # l_stance = -l_swing + 1
    # r_swing = np.flip(l_swing)
    # r_stance = -r_swing + 1

    fig, ax = plt.subplots(2, 1, figsize=(8,5))
    t = np.linspace(0, cycle_time, length)
    ax[0].plot(t, l_swing, label="swing")
    ax[0].plot(t, l_stance, label="stance")
    # ax[0].plot(t, l_force, label="l_force")
    ax[1].plot(t, r_swing, label="swing")
    ax[1].plot(t, r_stance, label="stance")
    # ax[1].plot(t, r_force, label="r_force")

    ax[0].scatter(t, l_swing, label="l_swing")
    ax[0].scatter(t, l_stance, label="l_stance")
    ax[1].scatter(t, r_swing, label="r_swing")
    ax[1].scatter(t, r_stance, label="r_stance")
    
    # print(r_stance[0], r_stance[-1])
    ax[0].set_title("Left Foot", fontsize=18)
    ax[1].set_title("Right Foot", fontsize=18)
    ax[1].set_xlabel("Time (sec)", fontsize=16)
    ax[0].set_ylabel("Cost Weighting", fontsize=16)
    ax[1].set_ylabel("Cost Weighting", fontsize=16)

    ax[0].legend(loc=1)
    ax[1].legend(loc=1)

    fig.suptitle("Foot Cost Clock", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.show()
    # plt.savefig("./linclock5.svg")

def plot_step_clock():
    time = 0.841
    length = 400
    t = np.linspace(0, time, length)
    l_swing = np.ones(length)
    l_swing[int(length/2):] = 0
    r_swing = np.zeros(length)
    r_swing[int(length/2):] = 1
    l_stance = 1 - l_swing
    r_stance = 1 - r_swing

    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    t = np.linspace(0, 0.841, length)
    ax[0].plot(t, l_swing, label="left swing")
    ax[0].plot(t, l_stance, label="left stance")
    ax[1].plot(t, r_swing, label="right swing")
    ax[1].plot(t, r_stance, label="right stance")
    ax[0].legend()
    ax[1].legend()
    ax[1].set_xlabel("Time (sec)", fontsize=14)
    ax[0].set_ylabel("Weighting", fontsize=14)
    ax[1].set_ylabel("Weighting", fontsize=14)
    ax[0].tick_params(axis='both', which='major', labelsize=14)
    ax[1].tick_params(axis='both', which='major', labelsize=14)
    fig.suptitle("Foot Cost Weighting", fontsize=18)
    # plt.tight_layout()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def animate_lin_clock5():
    phaselen = 30
    swing_ratio = 0.4
    length = 200
    num_cycles = 1
    cycle_time = 1
    phases = np.linspace(0, num_cycles*phaselen, length)
    l_swing = np.zeros(length)
    l_stance = np.zeros(length)
    r_swing = np.zeros(length)
    r_stance = np.zeros(length)
    l_force = np.zeros(length)
    r_force = np.zeros(length)
    for i in range(length):
        l_swing[i], r_swing[i] = lin_clock5(swing_ratio, phases[i], phaselen)
        l_force[i] = math.ceil(l_swing[i])
        r_force[i] = math.ceil(r_swing[i])
        l_stance[i] = -l_swing[i] + 1
        r_stance[i] = -r_swing[i] + 1

    fig, ax = plt.subplots(2, 1, figsize=(8,5))
    t = np.linspace(0, cycle_time, length)
    ax[0].plot(t, l_swing, label="swing")
    ax[0].plot(t, l_stance, label="stance")

    ax[1].plot(t, r_swing, label="swing")
    ax[1].plot(t, r_stance, label="stance")

    ax[0].set_title("Left Foot", fontsize=18)
    ax[1].set_title("Right Foot", fontsize=18)
    ax[1].set_xlabel("Time (sec)", fontsize=16)
    ax[0].set_ylabel("Cost Weighting", fontsize=16)
    ax[1].set_ylabel("Cost Weighting", fontsize=16)
    ax[0].legend(loc=1)
    ax[1].legend(loc=1)

    fig.suptitle("Foot Cost Clock", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    v1 = ax[0].axvline(0, ls='-', color='r', lw=1, zorder=10)
    v2 = ax[1].axvline(0, ls='-', color='r', lw=1, zorder=10)
    v1.set_xdata([0.5, 0.5])
    duration = 1 # in sec
    refreshPeriod = 10 # in ms

    def animate(i, v1, v2, frames):
        # print("i:", i)
        t = (i / frames) % 1
        # t = i*period / 1000
        # print("t:", t)
        v1.set_xdata([t, t])
        v2.set_xdata([t, t])
        return v1, v2, 

    num_frames = int(duration/(refreshPeriod/1000))
    ani = animation.FuncAnimation(fig, animate, frames=num_frames, fargs=(v1, v2, num_frames), interval=refreshPeriod, blit=False)
    # ani = animation.FuncAnimation(fig, animate, frames=120, interval=1/120*1000, blit=True, fargs=(save_count=50)


    # plt.show()
    writer = animation.FFMpegWriter(
        fps=60, bitrate=1800)
    ani.save("./clock_film.mp4", writer=writer)

# plot_smooth_clock()
# plot_clock()
# plot_lin_clock2()
# plot_lin_clock3()
# plot_lin_clock4()
plot_lin_clock5()
# animate_lin_clock5()
# plot_step_clock()
# start_t = time.time()
# l_swing, r_swing = lin_clock3(.5, 0, 30)
# print((time.time() - start_t)*1e9)

