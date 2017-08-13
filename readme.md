This is a small pytorch library that contains some implementations of
reinforcement learning algorithms. I use this for learning and for my research
which centers around continuous domains, which is why these algorithms do not
have discrete implementations, though it'd be trivial to change them.

Soon to be implemented:

* PPO
* A3C
* Parallelism
* Beta distribution policy

Bells and whistles:
* GAE and TD(lambda) estimators
* Variable step size for VPG (aka poor man's natural gradient)
* Entropy based exploration bonus
* Observation and advantage centering

To be implemented long term:

* DDPG
* SVG
* Noisy Nets for Exploration
* CUDA support (should be trivial but I don't have a GPU to test on currently)

Maybe implemented in future:

* TRPO
* DXNN
* ACER and other off-policy methods
* Model-based methods

In the pipeline:
* Considering Visdom for progress tracking
* Option to render random trajectories in seperate thread during training
