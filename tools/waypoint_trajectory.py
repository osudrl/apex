import pygame
from utils.elements import Mouse, Robot, Waypoint, Trajectory, Grid

from scipy.interpolate import make_interp_spline, BSpline, interp1d, PchipInterpolator, UnivariateSpline
from scipy.integrate import quad

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

import pickle

# -- 10m by 10m grid
SCREEN_HEIGHT = 1000
SCREEN_WIDTH = 1000
PX_2_M = 50 # X pixels = 1.0 meters
TIME_BETWEEN_WAYPOINTS = 5 # 5 seconds between waypoints
FREQUENCY = 30

def animated_plot(time_data, x_data, y_data):
    
    y_data = -y_data

    fig, ax = plt.subplots()
    x, y = [], []
    ax.set_xlim(np.amin(x_data), np.amax(x_data))
    ax.set_ylim(np.amin(y_data), np.amax(y_data))
    line, = ax.plot(x_data[0], y_data[0])
    point, = ax.plot(x_data[0], y_data[0], 'o')
    
    def animation_frame(t):
        x.append(x_data[t])
        y.append(y_data[t])
        line.set_xdata(x)
        line.set_ydata(y)
        point.set_xdata(x[-1])
        point.set_ydata(y[-1])
        return line, point,
    
    frames = np.arange(0, len(time_data), 1)
    animation = FuncAnimation(fig, func=animation_frame, frames=frames, interval=FREQUENCY, repeat=False)
    plt.show()



class World:
    def __init__(self,screen_width,screen_height,px_2_m):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.px_2_m = px_2_m
        self.background_color = (0,0,0)
        self.mouse = Mouse(self.px_2_m)
        self.robot = None
        self.grid = Grid(screen_width, screen_height, px_2_m)

        self.waypoints = []
        self.trajectory = None

    def place_waypoint(self):
        mouse_position = self.mouse.get_position()
        self.waypoints.append(Waypoint(mouse_position))

    # returns a spline of positions and angles
    def waypointTrajectory(self):

        frequency = FREQUENCY

        if len(self.waypoints) > 0:
            
            # xy points (position data)
            x = np.array([p.px for p in self.waypoints])
            y = np.array([p.py for p in self.waypoints])
            
            # parametrize by t
            t = np.linspace(0, 1, num=x.shape[0])
            t_new = np.linspace(0, 1, num=int(len(self.waypoints)*TIME_BETWEEN_WAYPOINTS*frequency))
            
            # pchip interpolation --- position
            pos_spl = PchipInterpolator(t, np.c_[x,y])
            x_new, y_new = pos_spl(t_new).T
            points = [list(t) for t in zip(x_new, y_new)]
            
            # get theta data
            theta_new = [np.arctan2((SCREEN_HEIGHT-y_new[i]) - (SCREEN_HEIGHT-y_new[i-1]), x_new[i] - x_new[i-1]) for i in range(1, len(points))]
            theta_new.append(0.0)

            # get velocity data with finite differencing
            # vel_spl = pos_spl.derivative()
            vx_new, vy_new = np.divide(np.diff(x_new), (1/frequency)), np.divide(np.diff(y_new), (1/frequency))
            # set final velocity to be zero
            vx_new = np.append(vx_new, 0.0)
            vy_new = np.append(vy_new, 0.0)
            # vx_new, vy_new = vel_spl(t_new).T
            # Mean Value Theorem to obtain average velocity between intervals
            # instant_vels = np.sqrt( np.power(vx_new, 2) + np.power(vy_new, 2))
            # vels = [abs((instant_vels[i] + instant_vels[i-1]) / 2)  for i in range(1, len(instant_vels))]
            # vels.append(0.0)
            vels = np.sqrt( np.power(vx_new, 2) + np.power(vy_new, 2))

            # print(vels)
            # input()

            # ax_new, ay_new = np.divide(np.diff(vx_new), (1/frequency)), np.divide(np.diff(vy_new), (1/frequency))
            # ax_new = np.append(ax_new, 0.0)
            # ay_new = np.append(ay_new, 0.0)
            # accel = np.sqrt( np.power(ax_new, 2) + np.power(ay_new, 2))
            # accel_spl = vel_spl.derivative()
            # accel = accel_spl(t_new)
            # animated_plot(t_new, x_new, y_new)
            # animated_plot(t_new, ax_new, ay_new)
            # plt.plot(t_new, accel, label="accel")
            # print(vels)
            # input()
            # vels = [np.sqrt(np.power(x_new[i] - x_new[i-1], 2) + np.power(y_new[i] - y_new[i-1], 2) / (1 / frequency) ) for i in range(1, len(points))]
            # vels.insert(0, 0.0)

            # vel_spl = pos_spl.derivative()
            # print(vel_spl(t_new))
            # xdot, ydot = vel_spl(t_new).T
            # vmag = np.sqrt(np.power(xdot, 2) + np.power(ydot, 2))
            # print((xdot, ydot))
            # input()
            # vels = [ (vmag[i] - vmag[i-1]) / (1 / frequency) for i in range(1, len(points))]
            # print("\nposes:\n{}\n\nthetas:\n{}\n\nvels:\n{}\n\n".format(points, theta_new, vels))

            # send to trajectory
            self.trajectory = Trajectory(t_new, points, theta_new, vels)
            if self.robot is not None:
                self.robot.trajectory = self.trajectory.positions
            return t_new, x_new, y_new, theta_new, vels, pos_spl
        return

    def initiate_robot(self, time_passed):
        self.robot = Robot(self.trajectory, time_passed, FREQUENCY)

    def bias_robot(self):
        mouse_position = self.mouse.get_position()
        self.robot.px = mouse_position[0]
        self.robot.py = mouse_position[1]

    def update(self,time_passed):
        mouse_position = self.mouse.get_position()
        self.mouse.update(time_passed)
        if self.robot is not None:
            self.robot.update(time_passed)

    def render(self,screen):
        screen.fill(self.background_color)
        self.grid.render(screen)
        self.mouse.render(screen)
        if self.robot is not None:
            self.robot.render(screen)
        [x.render(screen) for x in self.waypoints]
        if self.trajectory is not None:
            self.trajectory.render(screen)


def pygame_main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
    clock = pygame.time.Clock()
    world = World(SCREEN_WIDTH,SCREEN_HEIGHT,PX_2_M)
    elapsed_time = 0.0
    done_with_waypoints = False
    first_enter = True
    # Game loop:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return world.trajectory
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if not done_with_waypoints:
                    world.place_waypoint()
                    print("placed waypoint at {}".format(world.mouse.get_m_position()))
                else:
                    world.place_waypoint()
                    print("placed waypoint at {}".format(world.mouse.get_m_position()))
                    t_new, x_new, y_new, theta_new, vels, pos_spl = world.waypointTrajectory()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    if first_enter == True:
                        first_enter = False
                        t_new, x_new, y_new, theta_new, vels, pos_spl = world.waypointTrajectory()
                        print("generated trajectory")
                        done_with_waypoints = True
                    else:
                        world.robot.counter += 1
                elif event.key == pygame.K_v:
                    elapsed_time = 0.0
                    world.initiate_robot(elapsed_time)
                    print("robot")

        time_passed_seconds = clock.tick() / 1000.0
        # time_passed_seconds = clock.tick() / 100.0
        elapsed_time += time_passed_seconds

        # print(elapsed_time)

        world.update(elapsed_time)
        world.render(screen)

        pygame.display.update()

def main():

    trajectory = pygame_main()
    trajectory.prepare_for_export(PX_2_M, SCREEN_HEIGHT)

    command_trajectory = {"compos" : trajectory.positions,
                          "speed"  : trajectory.vels,
                          "orient" : trajectory.thetas}

    with open("command_trajectory.pkl", "wb") as f:
        pickle.dump(command_trajectory, f)
        print("wrote pickle file")

    np.savetxt('waypoints.csv', np.array(trajectory.positions)[:,0:2], delimiter=",")
    print("wrote waypoints to csv file. use add_waypoints.py script to visualize them in mujoco")
    print(" (ex.)  --  python add_waypoints.py -i \"../cassie/cassiemujoco/cassie.xml\" -o \"../cassie/cassiemujoco/cassie_waypoints.xml\" -w \"waypoints.csv\"")

if __name__ == "__main__":
    main()