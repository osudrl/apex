# Pytorch RL

This is a small, modular library that contains some implementations of continuous reinforcement learning algorithms. Fully compatible with OpenAI gym.


## Running experiments

### Basics
An example of running an experiment using vanilla policy gradient is shown in ```examples/vpg.py```.

```python
env = gym.make("Hopper-v1")

obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

policy = GaussianMLP(obs_dim, action_dim, (8,))
baseline = FeatureEncodingBaseline(obs_dim)

algo = VPG(
    env=env,
    policy=policy,
    baseline=baseline,
    lr=args.lr,
)
```
This script shows the basics of defining a model and setting up an algorithm. The recommended way to train the model, log diagnostics, and visualize the current policy all at once is to use the ```rl.utils.run_experiment()``` function.

```python
run_experiment(algo, args, log=True, render=True)
```

This sets up the logging, launches a progress monitor in your browser, renders the policy continuously in a seperate thread, and executes ```algo.train()``` according to your hyperparameters in ```args```.

### Logging details
For the logger to work properly, you want to supply all of your hyperparameters via command line arguments through argparser, and pass the resulting object to an instance of ```rl.utils.Logger```. Algorithms take care of their own diagnostic logging; if you don't wont to log anything, simply don't pass a logger object to ```algo.train()```.

Beyond your algorithm hyperparameters, the Logger expects that you supply an argument named ```logdir```, containing the root directory you want to store your logfiles in, and an argument named ```seed```, which is used to seed the pseudorandom number generators.

An basic command line script illustrating this is:
```bash
python examples/vpg.py --logdir experiments/ --seed l337
```

The resulting directory tree would look something like this:
```
experiments/
└── e2374a18f0
    ├── experiment.info
    └── seed1337.log
```
With ```e2374a18f0``` being a hash uniquely generated from the hyperparameter values used, ```experiment.info``` being a plaintext file listing those hyperparameters, and ```seed1337.log``` being a tab-seperated-value file containing the logged diagnostics (e.g. reward, kl divergence, etc) for each iteration.

This file structure makes it easy to compare models generated using the same hyperparameters but different seeds, and neatly seperates experiments by hyperparameter settings.

## Monitoring live training progress

### With bokeh
Calling ```run_experiment()``` with ```monitor=True``` automatically launches a bokeh instance. To manually monitor training progress in a browser using bokeh, run the following command, giving it the path to the active log file.
```bash
bokeh serve --show bokeh_monitor.py --args path/to/file.log
```
This should open a tab to http://localhost:5006/bokeh_monitor in your browser. If all goes well, you should see something like this image:

![alt-text](docs/bokeh_monitor.png)

### With matplotlib

Comming soon.


### To Do
- [ ] Make algorithms handle their own argument parsing
- [ ] Package everything
- [ ] Clean up /utils/
- [ ] Make baselines and models take in an env




## Soon to be implemented:

* PPO
* A3C
* Parallelism
* Beta distribution policy

Bells and whistles:
* GAE and TD(lambda) estimators
* Variable step size for VPG (aka poor man's natural gradient)
* Entropy based exploration bonus
* Observation and advantage centering
* Pytorch wrapper for OpenAI Gym environments

To be implemented long term:

* DDPG
* NAF
* SVG
* I2A
* PGPE?
* Noisy Nets for Exploration
* CUDA support (should be trivial but I don't have a GPU to test on currently)

Maybe implemented in future:

* TRPO
* DXNN
* ACER and other off-policy methods
* Model-based methods

Implemented:
* VPG plus + baseline confirmed to be correct and fast.
* No baseline implementation to test against, but adaptive VPG appears correct.
