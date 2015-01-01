#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Testing extensions to Sztorc consensus.

"""
from __future__ import division
import os
import sys
import platform
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
            votes = np.array([[  1,  1,  0,  1],
                              [  1,  0,  0,  0],
                              [  1,  1,  0,  0],
                              [  1,  1,  1,  0],
                              [  1,  0,  1,  1],
                              [  0,  0,  1,  1]])
            # new: true=1, false=-1, indeterminate=0.5, no response=0
            votes = np.array([[  1,  1, -1,  1],
                              [  1, -1, -1, -1],
                              [  1,  1, -1, -1],
                              [  1,  1,  1, -1],
                              [  1, -1,  1,  1],
                              [ -1, -1,  1,  1]])
            reputation = [2, 10, 4, 2, 7, 1]
            oracle = Oracle(votes=votes, reputation=reputation)
            pprint(oracle.consensus())
        elif opt in ('-m', '--missing'):
            # old: true=1, false=0, indeterminate=0.5, no response=-1
            votes = np.array([[  1,  1,  0, -1],
                              [  1,  0,  0,  0],
                              [  1,  1,  0,  0],
                              [  1,  1,  1,  0],
                              [ -1,  0,  1,  1],
                              [  0,  0,  1,  1]])
            votes = np.array([[      1,  1,  0, np.nan],
                              [      1,  0,  0,      0],
                              [      1,  1,  0,      0],
                              [      1,  1,  1,      0],
                              [ np.nan,  0,  1,      1],
                              [      0,  0,  1,      1]])
            # new: true=1, false=-1, indeterminate=0.5, no response=0
            votes = np.array([[  1,  1, -1,  0],
                              [  1, -1, -1, -1],
                              [  1,  1, -1, -1],
                              [  1,  1,  1, -1],
                              [  0, -1,  1,  1],
                              [ -1, -1,  1,  1]])
            # new: true=1, false=-1, indeterminate=0.5, no response=np.nan
            votes = np.array([[      1,  1, -1, np.nan],
                              [      1, -1, -1,     -1],
                              [      1,  1, -1,     -1],
                              [      1,  1,  1,     -1],
                              [ np.nan, -1,  1,      1],
                              [     -1, -1,  1,      1]])
            reputation = [2, 10, 4, 2, 7, 1]
            oracle = Oracle(votes=votes, reputation=reputation)
            pprint(oracle.consensus())
        elif opt in ('-t', '--test'):
            votes = np.array([[ 1, 0.5,  0,  0],
                              [ 1, 0.5,  0,  0],
                              [ 1,   1,  0,  0],
                              [ 1, 0.5,  0,  0],
                              [ 1, 0.5,  0,  0],
                              [ 1, 0.5,  0,  0],
                              [ 1, 0.5,  0,  0]])
