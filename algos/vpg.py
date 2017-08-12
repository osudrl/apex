"""
Implementation of vanilla policy gradient.
Current problems:
1) Way too slow. Likely bottleneck is storing/calculating an
   unneccessary amount of computation graphs via autograd variables
   Solution: cache observations then batch them so there's only one
   computation graph.
2) Weird bug is causing learning to fail catastrophically (where the gradient
   is too large?) causing massive fluctuations
3) No baseline/variance reduction. This is just sort of a "todo"
4) Doesn't use PT's ".reinforce()" idiom because I don't like abstracting
   away too many mathematical details. TODO: evaluate if this is still the most
   readable idiom after speed optimizations
5) No parallel sampling. Just another todo. Might have parallel algos
   reimplemented in their own area and keep a seperate more "readable" section
   for their sequential counterparts. I want to keep this code as minimal
   and clear as possible.
"""
import torch
import torch.optim as optim
from torch.autograd import Variable


class VPG():
    def __init__(self, env, policy, discount=0.99):
        self.env = env
        self.policy = policy
        #self.baseline = baseline
        self.discount = discount

        self.optimizer = optim.Adam(policy.parameters(), lr=0.01)

    def train(self, n_itr, n_trj, max_trj_len):
        env = self.env
        policy = self.policy
        for _ in range(n_itr):

            losses = []
            total_r = []
            for _ in range(n_trj):
                rewards = []

                obs = env.reset()

                logprobs = []
                for _ in range(max_trj_len):
                    obs_var = Variable(torch.Tensor(obs).unsqueeze(0))
                    means, log_stds, stds = policy(obs_var)

                    action = policy.get_action(means, stds)

                    logprobs.append(
                        policy.log_likelihood(action, means, log_stds, stds)
                    )

                    next_obs, reward, done, _ = env.step(action.data.numpy())
                    rewards.append(reward)

                    obs = next_obs

                    if done:
                        break

                total_r.append(sum(rewards))

                R = Variable(torch.zeros(1, 1))
                loss = 0
                for i in reversed(range(len(rewards))):
                    R = self.discount * R + rewards[i]
                    loss = loss - logprobs[i] * R

                losses.append(loss)

            print(sum(total_r) / len(total_r))

            policy_loss = sum(losses) / len(losses)
            print("loss: %s, num: %s" % (policy_loss.data.numpy(), len(losses)))

            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()
