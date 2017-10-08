import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import gym


class RLDataset(Dataset):
    def __init__(self, observations, returns):
        self.observations = observations
        self.returns = returns

    def __len__(self):
        return self.returns.numel()

    def __getitem__(self, idx):
        return (
            self.observations[idx],
            self.returns[idx]
        )


class Critic(nn.Module):

    def __init__(self, obs_dim, hidden_dims=(32,), nonlin=F.tanh):

        super(Critic, self).__init__()

        self.hidden_layers = nn.ModuleList()
        self.hidden_layers += [nn.Linear(obs_dim, hidden_dims[0])]

        for l in range(len(hidden_dims) - 1):
            in_dim = hidden_dims[l]
            out_dim = hidden_dims[l + 1]

            self.hidden_layers += [nn.Linear(in_dim, out_dim)]

        self.vf = nn.Linear(hidden_dims[-1], 1) # value function estimator

        self.nonlin = nonlin

    def forward(self, x):
        output = x
        for l in self.hidden_layers:
            output = self.nonlin(l(output))
        critic = self.vf(output)

        return critic


if __name__ == "__main__":
    env = gym.make("Hopper-v1")

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    critic = Critic(obs_dim)

    optimizer = Adam(
        list(critic.parameters()),
        lr=3e-4
    )

    for itr in range(10):
        print("======== %d ========" % itr)
        dataset = RLDataset(
            torch.rand(3000, obs_dim) * 100,
            torch.rand(3000, 1) * 100)

        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        for epoch in range(10):
            losses = []
            for batch in dataloader:
                obs, ret = map(Variable, batch)

                optimizer.zero_grad()

                c_loss = (critic(obs) - ret).pow(2).mean()

                #c_loss.backward()
                optimizer.step()

                losses.append(c_loss.data[0])

            print(losses)
            print(sum(losses)/len(losses))
