<img src="https://github.com/osudrl/apex/blob/master/apex-logo.png" alt="apex" width="200"/>

----

Apex is a small, modular library that contains some implementations of continuous reinforcement learning algorithms. Fully compatible with OpenAI gym.

## Running experiments

### Basics
An example of running an experiment using proximal policy optimization is shown in ```distributed_ppo.py```.

An example of running an experiment using twin-delayed deep deterministic policy gradient is shown in ```sync_td3.py```.

### Logging details / Monitoring live training progress
Tensorboard logging is enabled by default for all algorithms. The Logger expects that you supply an argument named ```logdir```, containing the root directory you want to store your logfiles in, and an argument named ```seed```, which is used to seed the pseudorandom number generators.

An basic command line script illustrating this is:
```bash
python distributed_ppo.py --logdir logs/ --seed l337
```

The resulting directory tree would look something like this:
```
logs/
├── ppo
│   └── experiments
│       └── [New Experiment Logdir]
├── synctd3
└── asynctd3
```

Using tensorboard makes it easy to compare experiments and resume training later on.

To see live training progress

Run ```$ tensorboard --logdir logs/ --port=8097``` then navigate to ```http://localhost:8097/``` in your browser

## Unit tests
You can run the unit tests using pytest.

### To Do
- [ ] Sphinx documentation and github wiki
- [ ] Make logger as robust and pythonic as possible
- [ ] Add experiments.info file into log directory like old visdom logger
- [ ] Fix some hacks having to do with support for parallelism 
(namely Vectorize, Normalize and Monitor)
- [ ] Improve/Tune implementations of TD3

### Notes

Troubleshooting: X module not found? Make sure PYTHONPATH is configured. Make sure you run 
examples from root directory.

## Features:
* Parallelism with [Ray](https://github.com/ray-project/ray)
* [GAE](https://arxiv.org/abs/1506.02438)/TD(lambda) estimators
* Variable step size for VPG (~roughly analagous to natural gradient, see PPO paper)
* Entropy based exploration bonus
* advantage centering (observation normalization WIP)
* [PPO](https://arxiv.org/abs/1707.06347), VPG with ratio objective and with log likelihood objective
* [TD3](https://arxiv.org/abs/1802.09477)
* [Parameter Noise Exploration](https://arxiv.org/abs/1706.01905) (for TD3 only)

#### To be implemented soon:

* Some form of massively parallel Evolutionary Algorithm as a baseline (CMAES, ARS)
* [A2C](https://arxiv.org/abs/1602.01783) 

#### To be implemented long term:
* [SAC](https://arxiv.org/abs/1801.01290)
* [GPO](https://arxiv.org/abs/1711.01012)
* [DDPG](https://arxiv.org/abs/1509.02971)
* [NAF](https://arxiv.org/abs/1603.00748)
* [SVG](https://arxiv.org/abs/1510.09142)
* [I2A](https://arxiv.org/abs/1707.06203)
* [PGPE](http://ieeexplore.ieee.org/document/5708821/?reload=true)
* [Value Distribution](https://arxiv.org/pdf/1707.06887.pdf)
* Oracle methods (e.g. [GPS](https://arxiv.org/abs/1610.00529))
* CUDA support (should be trivial but I don't have a GPU to test on currently)

#### Maybe implemented in future:

* [DXNN](https://arxiv.org/abs/1008.2412)
* [ACER](https://arxiv.org/abs/1611.01224) and other off-policy methods
* Model-based methods

## Acknowledgements

Thanks to @ikostrikov's whose great implementations I used to debug my own, and inspired a lot of changes to my original code. Also thanks to @rll for rllab, which inspired a lot of the high level interface and logging for this library, and to @OpenAI for the original PPO tensorflow implementation. Thanks to @sfujim for the original implementation of TD3 in PyTorch.