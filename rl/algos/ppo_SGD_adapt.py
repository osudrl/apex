"""Proximal Policy Optimization (clip objective)."""
from copy import deepcopy

import torch
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.autograd import Variable
from torch.distributions import kl_divergence
from torch.utils.tensorboard import SummaryWriter
from ..utils.logging import Logger

from rl.algos import PPO
from rl.envs import Vectorize, Normalize

import time

import numpy as np
import os

class PPO_SGD_adapt(PPO):

    def update_lr(self, optimizer, new_lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

    def train(self,
              env_fn,
              policy, 
              n_itr,
              normalize=None,
              logger=None):

        if normalize != None:
            policy.train()
        else:
            policy.train(0)

        env = Vectorize([env_fn]) # this will be useful for parallelism later
        
        if normalize is not None:
            env = normalize(env)

            mean, std = env.ob_rms.mean, np.sqrt(env.ob_rms.var + 1E-8)
            policy.obs_mean = torch.Tensor(mean)
            policy.obs_std = torch.Tensor(std)
            policy.train(0)

        env = Vectorize([env_fn])

        old_policy = deepcopy(policy)

        optimizer = optim.SGD(policy.parameters(), lr=self.lr)

        start_time = time.time()

        for itr in range(n_itr):
            print("********** Iteration {} ************".format(itr))

            sample_start = time.time()
            if self.n_proc > 1:
                print("doing multi samp")
                batch = self.sample_parallel(env_fn, policy, self.num_steps, 300)
            else:
                batch = self._sample(env_fn, policy, self.num_steps, 300) #TODO: fix this

            samp_time = time.time() - sample_start
            print("sample time: {:.2f} s".format(samp_time))

            observations, actions, returns, values = map(torch.Tensor, batch.get())

            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)

            minibatch_size = self.minibatch_size or advantages.numel()

            print("timesteps in batch: %i" % advantages.numel())

            old_policy.load_state_dict(policy.state_dict())  # WAY faster than deepcopy

            optimizer_start = time.time()

            for _ in range(self.epochs):
                losses = []
                sampler = BatchSampler(
                    SubsetRandomSampler(range(advantages.numel())),
                    minibatch_size,
                    drop_last=True
                )

                for indices in sampler:
                    indices = torch.LongTensor(indices)

                    obs_batch = observations[indices]
                    action_batch = actions[indices]

                    return_batch = returns[indices]
                    advantage_batch = advantages[indices]

                    values, pdf = policy.evaluate(obs_batch)

                    # TODO, move this outside loop?
                    with torch.no_grad():
                        _, old_pdf = old_policy.evaluate(obs_batch)
                        old_log_probs = old_pdf.log_prob(action_batch).sum(-1, keepdim=True)
                    
                    log_probs = pdf.log_prob(action_batch).sum(-1, keepdim=True)
                    
                    ratio = (log_probs - old_log_probs).exp()

                    cpi_loss = ratio * advantage_batch
                    clip_loss = ratio.clamp(1.0 - self.clip, 1.0 + self.clip) * advantage_batch
                    actor_loss = -torch.min(cpi_loss, clip_loss).mean()

                    critic_loss = 0.5 * (return_batch - values).pow(2).mean()

                    entropy_penalty = -self.entropy_coeff * pdf.entropy().mean()

                    # TODO: add ability to optimize critic and actor seperately, with different learning rates

                    optimizer.zero_grad()
                    (actor_loss + critic_loss + entropy_penalty).backward()
                    optimizer.step()

                    # Do adaptive step size to satisfy KL div threshold
                    with torch.no_grad():
                        _, pdf = policy.evaluate(obs_batch)
                    curr_lr = self.lr
                    while kl_divergence(pdf, old_pdf).mean() > 0.02:
                        curr_lr /= 2
                        self.update_lr(optimizer, curr_lr)
                        policy.load_state_dict(old_policy.state_dict())
                        optimizer.step()
                        with torch.no_grad():
                            _, pdf = policy.evaluate(obs_batch)

                    if curr_lr != self.lr:
                        print("KL div threshold violated, changed step size to ", curr_lr)
                        
                    losses.append([actor_loss.item(),
                                   pdf.entropy().mean().item(),
                                   critic_loss.item(),
                                   ratio.mean().item()])

                # TODO: add verbosity arguments to suppress this
                print(' '.join(["%g"%x for x in np.mean(losses, axis=0)]))

            opt_time = time.time() - optimizer_start

            if logger is not None:
                eval_start = time.time()
                eval_proc = min(self.n_proc, 24)
                test = self.sample(env, policy, 800 // eval_proc, 400, deterministic=True)
                eval_time = time.time() - eval_start
                print("evaluate time elapsed: {:.2f} s".format(eval_time))

                avg_eval_reward = np.mean(test.ep_returns)
                avg_batch_reward = np.mean(batch.ep_returns)
                avg_ep_len = np.mean(batch.ep_lens)
                _, pdf     = policy.evaluate(observations)
                _, old_pdf = old_policy.evaluate(observations)

                entropy = pdf.entropy().mean().item()
                kl = kl_divergence(pdf, old_pdf).mean().item()

                if type(logger) is Logger:
                    logger.record("Return (test)", avg_eval_reward)
                    logger.record("Return (batch)", avg_batch_reward)
                    logger.record("Mean Eplen",  avg_ep_len)
            
                    logger.record("Mean KL Div", kl)
                    logger.record("Mean Entropy", entropy)
                    logger.dump()
                elif type(logger) is SummaryWriter:

                    sys.stdout.write("-" * 37 + "\n")
                    sys.stdout.write("| %15s | %15s |" % ('Return (test)', avg_eval_reward) + "\n")
                    sys.stdout.write("| %15s | %15s |" % ('Return (batch)', avg_batch_reward) + "\n")
                    sys.stdout.write("| %15s | %15s |" % ('Mean Eplen', avg_ep_len) + "\n")
                    sys.stdout.write("| %15s | %15s |" % ('Mean KL Div', "%8.3g" % kl) + "\n")
                    sys.stdout.write("| %15s | %15s |" % ('Mean Entropy', "%8.3g" % entropy) + "\n")
                    sys.stdout.write("-" * 37 + "\n")
                    sys.stdout.flush()

                    logger.add_scalar("Data/Return (test)", avg_eval_reward)
                    logger.add_scalar("Data/Return (batch)", avg_batch_reward)
                    logger.add_scalar("Data/Mean Eplen", avg_ep_len)

                    logger.add_scalar("Misc/Mean KL Div", kl)
                    logger.add_scalar("Misc/Mean Entropy", entropy)
                    logger.add_scalar("Misc/Critic Loss", critic_loss, itr)
                    logger.add_scalar("Misc/Actor Loss", actor_loss, itr)

                    logger.add_histogram("Misc/Sample Times", samp_time, itr)
                    logger.add_histogram("Misc/Optimize Times", opt_time, itr)
                    logger.add_histogram("Misc/Evaluation Times", eval_time, itr)
                else:
                    print("No Logger")


            # TODO: add option for how often to save model
            # if itr % 10 == 0:
            if np.mean(test.ep_returns) > self.max_return:
                self.max_return = np.mean(test.ep_returns)
                self.save(policy, env)
                self.save_optim(optimizer)

            print("Total time: {:.2f} s".format(time.time() - start_time))


            
