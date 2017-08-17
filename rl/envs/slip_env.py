from rllab.envs.base import Env
from rllab.spaces import Box
from rllab.envs.base import Step

from rl.envs import pyslip as slip
import numpy as np
import pygame


def slip_step(s, dt, action):
    steps = state_to_np(slip.step(s, dt, action[0], action[1])), \
            slip.step(s, dt, action[0], action[1])
    return steps


def state_to_np(s):
    return np.array([s.x, s.y, s.phi, s.l, s.l_eq, s.theta, s.theta_eq, s.dx,
                     s.dy, s.dphi, s.dl, s.dl_eq, s.dtheta, s.dtheta_eq])


def convert_coord(x, y):
    return [int(100 * x) + 0*800//2, int(100 * -y) + 800//2]


class SlipEnv(Env):
    def __init__(self, dt):
        self.start_state = state_to_np(slip.reset())
        self.state = self.start_state
        self.pygame_init = False
        self.screen = None
        self.trj_len = 0
        self.last_state = self.state
        self.dt = dt
        self.cstate = slip.reset()
        contact = (self.cstate.l_eq - self.cstate.l) > 0.01 and self.cstate.dl < 0
        self.trunc_state = np.hstack((np.delete(self.state, [0, 1, 5, 12]), [contact]))
        self.last_trunc_state = self.trunc_state

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action.astype(float)  # correct weird float inconsistency
        return self._step(action, self.dt)

    def _step(self, action, dt):
        self.last_trunc_state = self.trunc_state
        self.last_state = self.state

        last_Fx = self.last_state[0] + self.last_state[3]*np.cos(-self.last_state[2]-self.last_state[5]+3.14/2)

        self.state, self.cstate = slip_step(self.state, dt, action)
        self.state = self.state.astype(float)

        curr_Fx = self.state[0] + self.state[3]*np.cos(-self.state[2]-self.last_state[5]+3.14/2)
        dFx = curr_Fx - last_Fx

        self.trj_len += 1

        # reward = s000 if self.state[1] < 0 else dFx + 5 * dX + dY
        reward = dFx #self.state[7] #- 10 * (self.state[0] < .3)# or self.state[0] > 1)

        #if self.state[1] <= 0:
        #    reward -= 100

        done = self.state[1] < 0.5 \
               or np.abs(self.state[5]) > 0.33 \
               or self.state[1] > 1 \
               or self.state[3] > 0.75 \
               or self.state[3] < 0.6

        if done:
            reward -= dFx#self.state[7]# + 3/self.state[1] + dFx


        contact = (self.cstate.l_eq - self.cstate.l) > 0.01 and self.cstate.dl < 0

        obs = np.hstack((np.delete(self.state, [0, 1, 5, 12]), [contact]))

        self.trunc_state = obs
        return Step(observation=obs,
                    reward=reward,
                    done=done)


    def reset(self):
        self.state = self.start_state

        #self.state[7] += np.random.normal(0, 0.1, 1)

        contact = (self.cstate.l_eq - self.cstate.l) > 0.01 and self.cstate.dl < 0
        obs = np.hstack((np.delete(self.state, [0, 1, 5, 12]), [contact])).astype(float)

        self.trj_len = 0
        return obs


    def render(self):
        if(not self.pygame_init):
            pygame.init()
            self.screen = pygame.display.set_mode([800, 800])
            self.pygame_init = True

        self.screen.fill((255, 255, 255))
        pygame.draw.line(self.screen, (0, 0, 20), convert_coord(-1e3, 0),
                         convert_coord(1e3, 0), 1)

        pygame.draw.circle(
            self.screen,
            (0, 0, 0),
            convert_coord(self.state[0], self.state[1]),
            10,
            1
        )

        pygame.draw.line(
            self.screen,
            (0, 0, 0),
            convert_coord(self.state[0], self.state[1]),
            convert_coord(self.state[0]+0.1*np.cos(self.state[2]),
                          self.state[1]+0.1*np.sin(self.state[2])),
            1
        )

        pygame.draw.line(
            self.screen,
            (0, 0, 0),
            convert_coord(self.state[0], self.state[1]),
            convert_coord(self.state[0]+self.state[3] *
                          np.cos(-self.state[2]-self.state[5]+3.14/2),
                          self.state[1]-self.state[3] *
                          np.sin(-self.state[2]-self.state[5]+3.14/2)),
            1
        )
        pygame.display.update()

    @property
    def action_space(self):
        return Box(low=-1e2, high=1e2, shape=(2,))

    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(11,))
