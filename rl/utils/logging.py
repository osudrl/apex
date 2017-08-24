"""
A simple live-updating logger for logging training progress.

Based largely off Berkely's DRL course HW4, which is itself inspired by rllab.
https://github.com/berkeleydeeprlcourse/homework/blob/master/hw4/logz.py
"""
from functools import partial
import os.path as osp, shutil, time, atexit, os, subprocess, hashlib, sys

from collections import OrderedDict

class Logger():
    def __init__(self, args):
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

        self.output_file = self._get_directory(args)
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
        active_path = osp.join(output_dir, "ACTIVE_" + filename)

        # Create a file with all the hyperparam settings in plaintext
        info_path = osp.join(output_dir, "experiment.info")
        self._generate_info_file(open(info_path, 'w'), arg_dict)

        print(self._colorize("Logging data to %s" % active_path,
                             'green', bold=True))

        rename = partial(
            os.rename,
            src=active_path,
            dst=osp.join(output_dir, filename)
        )

        # remove "ACTIVE_" prefix on program exit
        atexit.register(rename)

        return open(active_path, 'w')

    def record(self, key, val):
        """
        Log some diagnostic value in current iteration.

        Call this exactly once for each diagnostic, every iteration
        """
        if self.init:
            self.header.append(key)
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
            vals.append(val)
        sys.stdout.write("-" * 37 + "\n")
        sys.stdout.flush()

        if self.init:
            self.output_file.write("\t".join(self.header))
            self.output_file.write("\n")

        self.output_file.write("\t".join(map(str, vals)))
        self.output_file.write("\n")
        self.output_file.flush()

        self.current_row.clear()

        self.init = False

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
