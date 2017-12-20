from .render import *
from .experiment import *
from .plotting import *
from .logging import *
import sys

class ProgBar():
    def __init__(self, total, bar_len=40):
        self.total = total
        self.count = 0
        self.bar_len = bar_len

    def next(self, msg=''):
        self.count += 1

        fill_len = int(round(self.bar_len * self.count / float(self.total)))
        bar = '=' * fill_len + '-' * (self.bar_len - fill_len)

        percent = round(100.0 * self.count / float(self.total), 1)

        msg = msg.ljust(len(msg) + 2)

        sys.stdout.write('[%s] %s%s ... %s\r' % (bar, percent, '%', msg))
        sys.stdout.flush()
