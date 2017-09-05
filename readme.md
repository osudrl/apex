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
run_experiment(algo, args, log=True, render=True, monitor=True)
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
- [ ] Save models
- [ ] Merge CPI and VPG
- [ ] Package everything
- [ ] Clean up /utils/
- [ ] Make baselines and models take in an env
- [ ] use \_\_all\_\_ idiom


### Notes
I'm not satisfied with the semantic clarity of how distribution is encapsulated/abstracted. I'll probably change it to something less confusing to read.


## Implemented:
* [GAE](https://arxiv.org/abs/1506.02438)/TD(lambda) estimators
* Variable step size for VPG (~roughly analagous to natural gradient, see PPO paper)
* Entropy based exploration bonus
* advantage centering (observation normalization planned soon)
* [PPO](https://arxiv.org/abs/1707.06347), VPG with ratio objective and with log likelihood objective

### To be implemented soon:

* [A2C](https://arxiv.org/abs/1602.01783) 
* Parallelism
* [Beta distribution policy](http://proceedings.mlr.press/v70/chou17a/chou17a.pdf)
* [Parameter noise exploration](https://arxiv.org/abs/1706.01905)


### To be implemented long term:

* [DDPG](https://arxiv.org/abs/1509.02971)
* [NAF](https://arxiv.org/abs/1603.00748)
* [SVG](https://arxiv.org/abs/1510.09142)
* [I2A](https://arxiv.org/abs/1707.06203)
* [PGPE](http://ieeexplore.ieee.org/document/5708821/?reload=true)
* [Value Distribution](https://arxiv.org/pdf/1707.06887.pdf)
* Oracle methods (e.g. [GPS](https://arxiv.org/abs/1610.00529))
* CUDA support (should be trivial but I don't have a GPU to test on currently)

### Maybe implemented:

* [TRPO](https://arxiv.org/abs/1502.05477)
* [DXNN](https://arxiv.org/abs/1008.2412)
* [ACER](https://arxiv.org/abs/1611.01224) and other off-policy methods
* Model-based methods

