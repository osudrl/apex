#!/usr/bin/env python

import os, sys, time
from functools import partial
from rl.utils import Monitor
from threading import Thread

def monitor_loop(monitor, log_file, header):
    while True:
        where = log_file.tell()
        line = log_file.readline()
        if not line:
            time.sleep(1)
            log_file.seek(where)
        else:
            vals = line.rstrip('\n').split('\t')
            for i in range(len(vals)):
                print("%s: %s" % (header[i], vals[i]))
                monitor.update(header[i], vals[i])
            time.sleep(0.1)

monitor = Monitor()

log_file = open(sys.argv[1], 'r')

header = log_file.readline().rstrip('\n').split('\t')

for key in header:
    monitor.add_plot(key)

thread = Thread(target=partial(monitor_loop, monitor=monitor,
                               log_file=log_file, header=header))
thread.start()
