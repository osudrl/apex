import torch
from torch import Tensor
from torch.autograd import Variable as Var
from torch.utils.data import DataLoader

from data import SplitDataset
from controller import controller
from tqdm import trange

from progbar import ProgBar
import logging
import sys

class DAgger():
    def __init__(self, env, learner, expert):
        self.env = env
        self.expert = expert
        self.learner = learner

        self.logger = logging.getLogger("dagger")
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        self.logger.setLevel(logging.INFO)

    def train(self, dagger_itr, epochs, trj_len, log=True):
        env = self.env
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        X, y = Tensor(trj_len, obs_dim), Tensor(trj_len, action_dim)

        dagger_init = True
        for d in range(dagger_itr):
            X_new, y_new = Tensor(trj_len, obs_dim), Tensor(trj_len, action_dim)

            obs = env.reset()
            for t in range(trj_len):
                obs_torch = Var(Tensor(obs[None, :]), requires_grad=False)
                action = self.learner(obs_torch).data.numpy()[0]

                expert_action = controller(env.cstate, 0, 1, 0)
                if dagger_init:
                    obs, _, done, _ = env.step(expert_action)
                    X[t, :], y[t, :] = Tensor(obs), Tensor(expert_action)

                else:
                    obs, _, done, _ = env.step(action)

                    if done:
                        X_new, y_new = X_new[0:t, :], y_new[0:t, :]
                        break

                    X_new[t, :], y_new[t, :] = Tensor(obs), Tensor(expert_action)

            if not dagger_init:
                X, y = torch.cat((X, X_new), 0), torch.cat((y, y_new), 0)

            # TODO: add training class?
            BATCH_SIZE = 100
            dataset = SplitDataset(X.numpy(), y.numpy())
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

            bar = ProgBar(len(dataloader) * epochs)
            for e in range(epochs):
                running_loss = 0
                for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
                    X_batch, y_batch = Var(Tensor(X_batch)), Var(Tensor(y_batch))

                    running_loss += self.learner.fit(X_batch, y_batch)[0]
                    bar.next("DAgger iteration: %s / %s" % (d + 1, dagger_itr))

            dagger_init = False
