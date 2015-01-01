#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Testing extensions to Sztorc consensus.

"""
from __future__ import division
import os
import sys
import platform
from pprint import pprint
from colorama import Fore, Style, init
import numpy as np
import numpy.ma as ma
if platform.python_version() < "2.7":
    unittest = __import__("unittest2")
else:
    import unittest
from six.moves import xrange as range

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir))
sys.path.insert(0, os.path.join(HERE, os.pardir, "pyconsensus"))

from pyconsensus import Oracle, main

np.set_printoptions(linewidth=500)

tolerance = 1e-12
init()

def BR(string): # bright red
    return "\033[1;31m" + str(string) + "\033[0m"

def BB(string): # bright blue
    return Fore.BLUE + Style.BRIGHT + str(string) + Style.RESET_ALL

def BG(string): # bright green
    return Fore.GREEN + Style.BRIGHT + str(string) + Style.RESET_ALL

def blocky(*strings, **kwds):
    colored = kwds.get("colored", True)
    width = kwds.get("width", 108)
    bound = width*"#"
    fmt = "#{:^%d}#" % (width - 2)
    lines = [bound]
    for string in strings:
        lines.append(fmt.format(string))
    lines.append(bound)
    lines = "\n".join(lines)
    if colored:
        lines = BR(lines)
    return lines

if __name__ == "__main__":
    np.set_printoptions(linewidth=500)
    if argv is None:
        argv = sys.argv
    try:
        short_opts = 'hxm'
        long_opts = ['help', 'example', 'missing']
        opts, vals = getopt.getopt(argv[1:], short_opts, long_opts)
    except getopt.GetoptError as e:
        sys.stderr.write(e.msg)
        sys.stderr.write("for help use --help")
        return 2
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(__doc__)
            return 0
        elif opt in ('-x', '--example'):
            # old: true=1, false=0, indeterminate=0.5, no response=-1
            reports = np.array([[  1,  1,  0,  1],
                              [  1,  0,  0,  0],
                              [  1,  1,  0,  0],
                              [  1,  1,  1,  0],
                              [  1,  0,  1,  1],
                              [  0,  0,  1,  1]])
            # new: true=1, false=-1, indeterminate=0.5, no response=0
            reports = np.array([[  1,  1, -1,  1],
                              [  1, -1, -1, -1],
                              [  1,  1, -1, -1],
                              [  1,  1,  1, -1],
                              [  1, -1,  1,  1],
                              [ -1, -1,  1,  1]])
            reputation = [2, 10, 4, 2, 7, 1]
            oracle = Oracle(reports=reports, reputation=reputation)
            pprint(oracle.consensus())
        elif opt in ('-m', '--missing'):
            # old: true=1, false=0, indeterminate=0.5, no response=-1
            reports = np.array([[  1,  1,  0, -1],
                              [  1,  0,  0,  0],
                              [  1,  1,  0,  0],
                              [  1,  1,  1,  0],
                              [ -1,  0,  1,  1],
                              [  0,  0,  1,  1]])
            reports = np.array([[      1,  1,  0, np.nan],
                              [      1,  0,  0,      0],
                              [      1,  1,  0,      0],
                              [      1,  1,  1,      0],
                              [ np.nan,  0,  1,      1],
                              [      0,  0,  1,      1]])
            # new: true=1, false=-1, indeterminate=0.5, no response=0
            reports = np.array([[  1,  1, -1,  0],
                              [  1, -1, -1, -1],
                              [  1,  1, -1, -1],
                              [  1,  1,  1, -1],
                              [  0, -1,  1,  1],
                              [ -1, -1,  1,  1]])
            # new: true=1, false=-1, indeterminate=0.5, no response=np.nan
            reports = np.array([[      1,  1, -1, np.nan],
                              [      1, -1, -1,     -1],
                              [      1,  1, -1,     -1],
                              [      1,  1,  1,     -1],
                              [ np.nan, -1,  1,      1],
                              [     -1, -1,  1,      1]])
            reputation = [2, 10, 4, 2, 7, 1]
            oracle = Oracle(reports=reports, reputation=reputation)
            pprint(oracle.consensus())
        elif opt in ('-t', '--test'):
            reports = np.array([[ 1, 0.5,  0,  0],
                              [ 1, 0.5,  0,  0],
                              [ 1,   1,  0,  0],
                              [ 1, 0.5,  0,  0],
                              [ 1, 0.5,  0,  0],
                              [ 1, 0.5,  0,  0],
                              [ 1, 0.5,  0,  0]])
