import pygame
from pygame.locals import *

import math
import numpy as np


class Mouse:
    def __init__(self, px_2_m):
        self.px = 0
        self.py = 0
        self.vx = 0
        self.vy = 0
        self.radius = 0
        self.color = (100,200,100)
        self.px_2_m = px_2_m

    def get_position(self):
        return (self.px, self.py)
    
    def get_m_position(self):
        return (self.px / self.px_2_m, self.py / self.px_2_m)

    def get_velocity(self):
        return (self.vx, self.vy)
    
    def update(self, time_passed):
        prev_p = self.get_position()
        self.px, self.py  = pygame.mouse.get_pos()
        if time_passed > 0:
            self.vx = (self.px - prev_p[0]) / time_passed
            self.vy = (self.py - prev_p[1]) / time_passed
    
    def render(self, screen):
        pygame.draw.circle(screen, self.color, (self.px, self.py), self.radius)
    
class Robot:
    def __init__(self, trajectory, time_passed, frequency):
        
        # action space is forward velocity and heading
        self.positions = trajectory.positions
        self.velocities = trajectory.vels
        self.thetas = trajectory.thetas
        self.accels = trajectory.accels
        self.trajlen = len(trajectory.positions)
        
        # ground truth's position:
        self.t_px = int(self.positions[0][0])
        self.t_py = int(self.positions[0][1])

        # follower's pos
        self.f_px = int(self.positions[0][0])
        self.f_py = int(self.positions[0][1])

        self.radius = 10
        self.color = (50,50,200) # direct position tracker
        self.color2 = (200,50,50)  # velocity + angle tracker
        
        self.frequency = frequency
        self.prev_time = self.prev_inc_time = time_passed
        self.counter = 0
        self.count_inc = 1

    def update(self,time_passed):

        curr_accel = self.accels[self.counter]
        curr_vel = self.velocities[self.counter]
        curr_theta = self.thetas[self.counter]
        track_pos = self.positions[self.counter]

        # print((curr_vel, curr_theta, np.cos(curr_theta), np.sin(curr_theta)))

        # ground truth's new position:
        self.t_px, self.t_py = track_pos[0], track_pos[1]

        # follower's new position: execute angle and velocity command for time passed
        t_diff = time_passed - self.prev_time
        vx, vy = curr_vel * np.cos(curr_theta), curr_vel * np.sin(curr_theta)
        ax, ay = curr_accel * np.cos(curr_theta), curr_accel * np.sin(curr_theta)
        # gotta subtract the y velocity add because pygame counts y from top down
        self.f_px, self.f_py = self.f_px + vx * t_diff + 0.5 * ax * t_diff**2, self.f_py - vy * t_diff + 0.5 * ay * t_diff**2
        # self.f_px, self.f_py = self.f_px + vx * t_diff, self.f_py - vy * t_diff

        # increment t_idx on 30 Hz cycle
        if time_passed - self.prev_inc_time > (1 / self.frequency):
            self.counter += 1
            self.prev_inc_time = time_passed

        self.prev_time = time_passed

        # check if we need to restart
        if self.counter == self.trajlen:
            self.counter = 0
            self.f_px, self.f_py = int(self.positions[0][0]),int(self.positions[0][1])

    def return_info(self, px_2_m):

        # thetas are the yaw angle of the robot
        thetas_rotated = self.thetas # no rotation for now
        # center of mass position is x y position converted to meters, with constant z height
        positions_in_meters = np.array( [[self.trajectory[i][0] / px_2_m - self.trajectory[0][0] / px_2_m, self.trajectory[i][1] / px_2_m - self.trajectory[0][1] / px_2_m, 1.0] for i in range(len(self.trajectory))] )
        velocities_in_meters = np.array( [self.velocities[i] / px_2_m for i in range(len(self.velocities))] )

        print("positions:\n{}\n\nvelocities:\n{}\n\norient:\n{}\n".format(positions_in_meters, velocities_in_meters, thetas_rotated))

        return positions_in_meters, velocities_in_meters, thetas_rotated

    def render(self,screen):
        pygame.draw.circle(screen,self.color,(int(self.t_px),int(self.t_py)),self.radius)
        pygame.draw.circle(screen,self.color2,(int(self.f_px),int(self.f_py)),self.radius)
        # pygame.transform.rotate(screen, np.radians(self.theta))

class Waypoint:
    def __init__(self, mouse_position):
        self.px = mouse_position[0]
        self.py = mouse_position[1]
        self.radius = 5
        self.color = (100,200,100)

    def get_position(self):
        return (self.px, self.py)
    
    def render(self, screen):
        pygame.draw.circle(screen, self.color, (self.px, self.py), self.radius)

class Trajectory:
    def __init__(self, t_new, positions, thetas, vels, accels):
        self.param = t_new
        self.positions = positions
        self.thetas = thetas
        self.vels = vels
        self.accels = accels
        self.width = 2
        self.color = (100,200,100)
        self.arrow_color = (200,200,200)
        self.arrow_length = 20.0
    
    def render(self, screen):
        scaled_vels = self.vels / np.max(self.vels) * self.arrow_length
        pygame_poses = []
        for i in range(len(self.positions)):
            # pygame.draw.aaline(screen, self.color, self.positions[i-1], self.positions[i])
            # print(self.positions[i])
            pygame_poses.append((int(self.positions[i][0]), int(self.positions[i][1])))
            # circle for pos
            pygame.draw.circle(screen, self.color, pygame_poses[-1], self.width)
        for i in range(len(self.thetas)):
            # calculate next pos
            pos2 = (pygame_poses[i][0] + scaled_vels[i] * np.cos(self.thetas[i]) , pygame_poses[i][1] - scaled_vels[i] * np.sin(self.thetas[i]))
            # arrow for angle and vel
            pygame.draw.line(screen, self.arrow_color, pygame_poses[i], pos2)
    
    def prepare_for_export(self, scale_factor, screen_height):

        self.positions = [[self.positions[i][0] / scale_factor, (screen_height - self.positions[i][1]) / scale_factor, 1.0] for i in range(len(self.positions))]
        self.positions = [[self.positions[i][0]-self.positions[0][0], self.positions[i][1]-self.positions[0][1], self.positions[i][2]] for i in range(len(self.positions))]

        self.vels = [self.vels[i] / scale_factor for i in range(len(self.vels))]

        print("positions:\n{}\n\nvelocities:\n{}\n\norient:\n{}\n".format(self.positions[:5], self.vels[:5], self.thetas[:5]))
        print("max vel: {}".format(np.max(self.vels)))

class Grid:
    def __init__(self, screen_width, screen_height, px_2_m):
        self.px_2_m = px_2_m
        self.screen_height = screen_height
        self.screen_width = screen_width
        self.cell_height = px_2_m  # approx height of 1m x 1m cell
        self.cell_width = px_2_m   # approx width of 1m x 1m cell
        self.color = (90,90,90)
    
    def render(self, screen):
        # draw vertical lines
        for x in range(self.screen_height // self.px_2_m):
            pygame.draw.line(screen, self.color, (x * self.cell_width,0), (x * self.cell_width,self.screen_height))
        # draw horizontal lines
        for y in range(self.screen_width // self.px_2_m):
            pygame.draw.line(screen, self.color, (0, y * self.cell_height), (self.screen_width, y * self.cell_height))
