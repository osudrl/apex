#!/usr/bin/env python

import os, sys, time
from functools import partial
from threading import Thread

from rl.utils import Monitor

def monitor_loop(monitor, log_file, header):
    while True:
        where = log_file.tell()
        line = log_file.readline()
        if not line:
            time.sleep(.5)
            log_file.seek(where)
        else:
            vals = line.rstrip('\n').split('\t')
            for i in range(len(vals)):
                monitor.update(header[i], vals[i])
            time.sleep(0.2)

monitor = Monitor()

log_file = open(sys.argv[1], 'r')

header = log_file.readline().rstrip('\n').split('\t')

for key in header:
    monitor.add_plot(key)

thread = Thread(target=partial(monitor_loop, monitor=monitor,
                               log_file=log_file, header=header))
thread.start()
