import time
from copy import deepcopy
import torch
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import kl_divergence

import numpy as np
from rl.algos import PPO

# TODO:
# env.mirror() vs env.matrix?

# TODO: use magic to make this reuse more code (callbacks etc?)

class MirrorPPO(PPO):
    def update(self, policy, old_policy, optimizer,
               observations, actions, returns, advantages,
               env_fn
    ):
        env = env_fn()
        mirror_observation = env.mirror_observation
        if env.clock_based:
            mirror_observation = env.mirror_clock_observation
        mirror_action = env.mirror_action

        minibatch_size = self.minibatch_size or advantages.numel()

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
                    # obs_batch = torch.cat(
                    #     [obs_batch,
                    #      obs_batch @ torch.Tensor(env.obs_symmetry_matrix)]
                    # ).detach()

                    action_batch = actions[indices]
                    # action_batch = torch.cat(
                    #     [action_batch,
                    #      action_batch @ torch.Tensor(env.action_symmetry_matrix)]
                    # ).detach()

                    return_batch = returns[indices]
                    # return_batch = torch.cat(
                    #     [return_batch,
                    #      return_batch]
                    # ).detach()

                    advantage_batch = advantages[indices]
                    # advantage_batch = torch.cat(
                    #     [advantage_batch,
                    #      advantage_batch]
                    # ).detach()

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

                    # Mirror Symmetry Loss
                    _, deterministic_actions = policy(obs_batch)
                    if env.clock_based:
                        mir_obs = mirror_observation(obs_batch, env.clock_inds)
                        _, mirror_actions = policy(mir_obs)
                    else: 
                        _, mirror_actions = policy(mirror_observation(obs_batch))
                    mirror_actions = mirror_action(mirror_actions)

                    mirror_loss = 4 * (deterministic_actions - mirror_actions).pow(2).mean()

                    entropy_penalty = -self.entropy_coeff * pdf.entropy().mean()

                    # TODO: add ability to optimize critic and actor seperately, with different learning rates

                    optimizer.zero_grad()
                    (actor_loss + critic_loss + mirror_loss + entropy_penalty).backward()

                    # Clip the gradient norm to prevent "unlucky" minibatches from 
                    # causing pathalogical updates
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), self.grad_clip)
                    optimizer.step()

                    losses.append([actor_loss.item(),
                                   pdf.entropy().mean().item(),
                                   critic_loss.item(),
                                   ratio.mean().item(),
                                   mirror_loss.item()])

                # TODO: add verbosity arguments to suppress this
                print(' '.join(["%g"%x for x in np.mean(losses, axis=0)]))

                # Early stopping 
                if kl_divergence(pdf, old_pdf).mean() > 0.02:
                    print("Max kl reached, stopping optimization early.")
                    break

    def train(self,
              env_fn,
              policy, 
              n_itr,
              logger=None):

        old_policy = deepcopy(policy)

        optimizer = optim.Adam(policy.parameters(), lr=self.lr, eps=self.eps)

        start_time = time.time()

        for itr in range(n_itr):
            print("********** Iteration {} ************".format(itr))

            sample_start = time.time()
            batch = self.sample_parallel(env_fn, policy, self.num_steps, self.max_traj_len)

            print("time elapsed: {:.2f} s".format(time.time() - start_time))
            print("sample time elapsed: {:.2f} s".format(time.time() - sample_start))

            observations, actions, returns, values = map(torch.Tensor, batch.get())

            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)

            minibatch_size = self.minibatch_size or advantages.numel()

            print("timesteps in batch: %i" % advantages.numel())

            old_policy.load_state_dict(policy.state_dict())  # WAY faster than deepcopy

            optimizer_start = time.time()

            self.update(policy, old_policy, optimizer, observations, actions, returns, advantages, env_fn) 
           
            print("optimizer time elapsed: {:.2f} s".format(time.time() - optimizer_start))        


            if logger is not None:
                evaluate_start = time.time()
                test = self.sample_parallel(env_fn, policy, 800 // self.n_proc, self.max_traj_len, deterministic=True)
                print("evaluate time elapsed: {:.2f} s".format(time.time() - evaluate_start))

                _, pdf     = policy.evaluate(observations)
                _, old_pdf = old_policy.evaluate(observations)

                entropy = pdf.entropy().mean().item()
                kl = kl_divergence(pdf, old_pdf).mean().item()

                logger.record('Return (test)', np.mean(test.ep_returns), itr, 'Return', x_var_name='Iterations', split_name='test')
                logger.record('Return (batch)', np.mean(batch.ep_returns), itr, 'Return', x_var_name='Iterations', split_name='batch')
                logger.record('Mean Eplen', np.mean(batch.ep_lens), itr, 'Mean Eplen', x_var_name='Iterations', split_name='batch')

                logger.record('Mean KL Div', kl, itr, 'Mean KL Div', x_var_name='Iterations', split_name='batch')
                logger.record('Mean Entropy', entropy, itr, 'Mean Entropy', x_var_name='Iterations', split_name='batch')

                logger.dump()

            # TODO: add option for how often to save model
            if np.mean(test.ep_returns) > self.max_return:
                self.max_return = np.mean(test.ep_returns)
                self.save(policy)

            print("Total time: {:.2f} s".format(time.time() - start_time))
