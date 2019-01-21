"""
A simple live-updating logger for logging training progress.

Based largely off Berkely's DRL course HW4, which is itself inspired by rllab.
https://github.com/berkeleydeeprlcourse/homework/blob/master/hw4/logz.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from functools import partial
import os.path as osp, shutil, time, atexit, os, subprocess, hashlib, sys
import configparser
from collections import OrderedDict
import numpy as np

matplotlib.rcParams.update({'font.size': 8})

#from scipy.signal import medfilt

class Logger():
    def __init__(self, args, viz=True, viz_list=[]):
        self.ansi = dict(
            gray=30,
            red=31,
            green=32,
            yellow=33,
            blue=34,
            magenta=35,
            cyan=36,
            white=37,
            crimson=38
        )

        self.name = args.name

        self.viz_list = ["all"]

        self.args = args

        if viz:
            from visdom import Visdom
            self.viz = Visdom()
            self.wins = []
            self.viz_config = self.config_monitor()
        else:
            self.viz = None

        self.output_dir = self._get_directory(args)
        self.init = True
        self.header = []
        self.current_row = {}

    def _get_directory(self, args):
        """Use hyperparms to set a directory to output diagnostic files."""
        #get hyperparameters as dictionary
        arg_dict = args.__dict__

        assert "seed" in arg_dict, \
        "You must provide a 'seed' key in your command line arguments"

        assert "logdir" in arg_dict, \
        "You must provide a 'logdir' key in your command line arguments."

        #sort the keys so the same hyperparameters will always have the same hash
        arg_dict = OrderedDict(sorted(arg_dict.items(), key=lambda t: t[0]))

        #remove seed so it doesn't get hashed, store value for filename
        # same for logging directory
        seed = str(arg_dict.pop("seed"))
        logdir = str(arg_dict.pop('logdir'))

        # get a unique hash for the hyperparameter settings, truncated at 10 chars
        arg_hash = hashlib.md5(str(arg_dict).encode('ascii')).hexdigest()[0:10]

        output_dir = osp.join(logdir, arg_hash)

        # create a directory with the hyperparm hash as its name, if it doesn't
        # already exist.
        os.makedirs(output_dir, exist_ok=True)

        # create a file for this seed, this is where output will be logged
        filename = "seed" + seed + ".log"

        # currently logged-to directories will be pre-pended with "ACTIVE_"
        active_path = osp.join(output_dir, filename)

        # Create a file with all the hyperparam settings in plaintext
        info_path = osp.join(output_dir, "experiment.info")
        self._generate_info_file(open(info_path, 'w'), arg_dict)

        print(self._colorize("Logging data to %s" % active_path,
                             'green', bold=True))

        return active_path

    def record(self, key, val):
        """
        Log some diagnostic value in current iteration.

        Call this exactly once for each diagnostic, every iteration
        """
        if self.init:
            self.header.append(key)
            if self.viz is not None:
                self.wins.append(None)
        else:
            assert key in self.header, \
            "Key %s not in header. All keys must be set in first iteration" % key

        assert key not in self.current_row, \
        "You already set key %s this iteration. Did you forget to call dump()?" % key

        self.current_row[key] = val

    def dump(self):
        """Write all of the diagnostics from the current iteration"""
        vals = []

        sys.stdout.write("-" * 37 + "\n")
        for key in self.header:
            val = self.current_row.get(key, "")
            if hasattr(val, "__float__"):
                valstr = "%8.3g" % val
            else:
                valstr = val

            sys.stdout.write("| %15s | %15s |" % (key, valstr) + "\n")
            vals.append(float(val))
        sys.stdout.write("-" * 37 + "\n")
        sys.stdout.flush()

        output_file = None
        if self.init:
            output_file = open(self.output_dir, "w")

            output_file.write("\t".join(self.header))
            output_file.write("\n")
        else:
            output_file = open(self.output_dir, "a")

        output_file.write("\t".join(map(str, vals)))
        output_file.write("\n")
        output_file.flush()
        output_file.close()

        self.current_row.clear()

        self.init = False

        if self.viz is not None:
            self.plot()

    def config_monitor(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "../config/monitor.ini")

        config = configparser.ConfigParser()
        config.read(config_path)

        return config["monitor"]

    def plot(self):
        def running_mean(x, N):
            if len(x) < N:
                return x
            cumsum = np.cumsum(np.insert(x, 0, 0)) 
            return (cumsum[N:] - cumsum[:-N]) / float(N)

        data, header = self._load_data()

        for i in range(len(header)):
            y = data[:, i]

            if header[i] in self.viz_list or self.viz_list[0] == "all":
                # do some kind of window based smoothing
                #y = running_mean(y, 30) 

                xscale = 1 if self.viz_config["xlabel"] == "Iterations" \
                        else (self.args.num_steps / 1e6)

                x = np.arange(y.size) * xscale

                fig = plt.figure(figsize=(5,4))

                plt.plot(x, y, "C%i" % i)

                if self.viz_config["xlim"] == "Fixed":
                    plt.xlim(0, self.args.n_itr * xscale)

                plt.ylabel(header[i])
                plt.xlabel(self.viz_config["xlabel"])
                plt.title("{0}: {1}".format(header[i], self.name))

                plt.show()
                plt.draw()

                image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
                image = np.transpose(image, (2, 0, 1))

                plt.close(fig)

                self.wins[i] = self.viz.image(image, win=self.wins[i])

    def _load_data(self):
        log_file = open(self.output_dir, 'r')

        header = log_file.readline().rstrip('\n').split('\t')

        data = []
        for line in log_file:
            vals = line.rstrip('\n').split('\t')
            vals = [float(val) for val in vals]
            data.append(vals)
        
        data = np.array(data)

        log_file.close()
        return data, header

    def _generate_info_file(self, file, arg_dict):
        for key, val in arg_dict.items():
            file.write("%s: %s" % (key, val))
            file.write('\n')

    def _colorize(self, string, color, bold=False, highlight=False):
        """Format string to print with color 'color' if printed to unix terminal."""
        attr = []
        num = self.ansi[color]
        if highlight:
            num += 10
        attr.append(str(num))
        if bold:
            attr.append('1')
        return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)
