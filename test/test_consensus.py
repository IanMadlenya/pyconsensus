#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Consensus mechanism unit tests.

"""
from __future__ import division, unicode_literals, absolute_import
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

class TestConsensus(unittest.TestCase):

    def setUp(self):
        self.reports = [[1, 1, 0, 0],
                        [1, 0, 0, 0],
                        [1, 1, 0, 0],
                        [1, 1, 1, 0],
                        [0, 0, 1, 1],
                        [0, 0, 1, 1]]
        self.oracle = Oracle(reports=self.reports)
        self.c = [1, 2, 3, np.nan, 3]
        self.c2 = ma.masked_array(self.c, np.isnan(self.c))
        self.c3 = [2, 3, -1, 4, 0]

    def test_consensus(self):
        outcome = self.oracle.consensus()
        self.assertAlmostEquals(outcome["certainty"], 0.228237569613, places=11)

    def test_consensus_verbose(self):
        self.oracle = Oracle(reports=self.reports, verbose=True)
        outcome = self.oracle.consensus()
        self.assertAlmostEquals(outcome["certainty"], 0.228237569613, places=11)

    def test_consensus_weighted(self):
        reputation = np.array([1, 1, 1, 1, 1, 1])
        oracle = Oracle(reports=self.reports, reputation=reputation)
        outcome = oracle.consensus()
        self.assertTrue(0 <= outcome["certainty"] <= 1)
        self.assertTrue(0 <= outcome["participation"] <= 1)
        self.assertAlmostEquals(outcome["certainty"], 0.228237569612, places=11)

    def test_consensus_nans(self):
        reports = np.array([[1, 1, 0, 0],
                            [1, 0, 0, 0],
                            [1, 1, np.nan, 0],
                            [1, 1, 1, 0],
                            [0, 0, 1, 1],
                            [0, 0, 1, 1]])
        oracle = Oracle(reports=reports)
        outcome = oracle.consensus()
        self.assertTrue(0 <= outcome["certainty"] <= 1)
        self.assertTrue(0 <= outcome["participation"] <= 1)
        self.assertAlmostEquals(outcome["certainty"], 0.28865265952, places=11)

    def test_consensus_weighted_nans(self):
        reports = np.array([[1, 1, 0, 0],
                            [1, 0, 0, 0],
                            [1, 1, np.nan, 0],
                            [1, 1, 1, 0],
                            [0, 0, 1, 1],
                            [0, 0, 1, 1]])
        reputation = np.array([1, 1, 1, 1, 1, 1])
        oracle = Oracle(reports=reports, reputation=reputation)
        outcome = oracle.consensus()
        # print(outcome["Agents"]["OldRep"])
        # print(outcome["Agents"]["ThisRep"])
        self.assertTrue(0 <= outcome["certainty"] <= 1)
        self.assertTrue(0 <= outcome["participation"] <= 1)
        self.assertAlmostEquals(outcome["certainty"], 0.28865265952, places=11)

    def test_consensus_scaled(self):
        reports = [[ 0.3, 0.2, 0, 0],
                   [ 0.5, 0.3, 0, 0],
                   [ 0.4, 0.1, 0, 0],
                   [ 0.2, 0.7, 1, 0],
                   [ 0.1, 0.3, 1, 1],
                   [0.15, 0.2, 1, 1]]
        scalar_event_params = [
            {"scaled": True, "min": 0.1, "max": 0.5},
            {"scaled": True, "min": 0.2, "max": 0.7},
            {"scaled": False, "min": 0, "max": 1},
            {"scaled": False, "min": 0, "max": 1},
        ]
        oracle = Oracle(reports=reports, event_bounds=scalar_event_params)
        outcome = oracle.consensus()
        self.assertTrue(0 <= outcome["certainty"] <= 1)
        self.assertTrue(0 <= outcome["participation"] <= 1)
        self.assertAlmostEquals(outcome["certainty"], 0.362414826111, places=11)

    def test_consensus_scaled_nans(self):
        reports = np.array([[ 0.3, 0.2, 0, 0],
                            [ 0.5, 0.3, np.nan, 0],
                            [ 0.4, 0.1, 0, 0],
                            [ 0.2, 0.7, 1, 0],
                            [ 0.1, 0.3, 1, 1],
                            [0.15, 0.2, 1, 1]])
        scalar_event_params = [
            {"scaled": True, "min": 0.1, "max": 0.5},
            {"scaled": True, "min": 0.2, "max": 0.7},
            {"scaled": False, "min": 0, "max": 1},
            {"scaled": False, "min": 0, "max": 1},
        ]
        oracle = Oracle(reports=reports, event_bounds=scalar_event_params)
        outcome = oracle.consensus()
        self.assertTrue(0 <= outcome["certainty"] <= 1)
        self.assertTrue(0 <= outcome["participation"] <= 1)
        self.assertAlmostEquals(outcome["certainty"], 0.384073052730, places=11)
        # self.assertAlmostEquals(outcome["certainty"], 0.362414826111, places=11)

    def test_consensus_weighted_scaled_nans(self):
        reports = np.array([[ 0.3, 0.2, 0, 0],
                            [ 0.5, 0.3, np.nan, 0],
                            [ 0.4, 0.1, 0, 0],
                            [ 0.2, 0.7, 1, 0],
                            [ 0.1, 0.3, 1, 1],
                            [0.15, 0.2, 1, 1]])
        scalar_event_params = [
            {"scaled": True, "min": 0.1, "max": 0.5},
            {"scaled": True, "min": 0.2, "max": 0.7},
            {"scaled": False, "min": 0, "max": 1},
            {"scaled": False, "min": 0, "max": 1},
        ]
        oracle = Oracle(reports=reports,
                        event_bounds=scalar_event_params,
                        reputation=np.array([1] * 6))
        outcome = oracle.consensus()
        self.assertTrue(0 <= outcome["certainty"] <= 1)
        self.assertTrue(0 <= outcome["participation"] <= 1)
        self.assertAlmostEquals(outcome["certainty"], 0.384073052730, places=11)
        # self.assertAlmostEquals(outcome["certainty"], 0.362414826111, places=11)

    def test_consensus_array(self):
        oracle = Oracle(reports=np.array(self.reports))
        outcome = oracle.consensus()
        self.assertTrue(0 <= outcome["certainty"] <= 1)
        self.assertTrue(0 <= outcome["participation"] <= 1)
        self.assertAlmostEquals(outcome["certainty"], 0.228237569613, places=11)

    def test_consensus_masked_array(self):
        oracle = Oracle(reports=ma.masked_array(self.reports, np.isnan(self.reports)))
        outcome = oracle.consensus()
        self.assertTrue(0 <= outcome["certainty"] <= 1)
        self.assertTrue(0 <= outcome["participation"] <= 1)
        self.assertAlmostEquals(outcome["certainty"], 0.228237569613, places=11)

    def test_catch(self):
        expected = [0, 1, 0.5, 0]
        actual = [self.oracle.catch(0.4),
                  self.oracle.catch(0.6),
                  Oracle(reports=self.reports, catch_tolerance=0.3).catch(0.4),
                  Oracle(reports=self.reports, catch_tolerance=0.1).catch(0.4)]
        self.assertEqual(actual, expected)

    def test_weighted_prin_comp(self):
        expected = np.array([-0.81674714,
                             -0.35969107,
                             -0.81674714,
                             -0.35969107,
                              1.17643821,
                              1.17643821])
        actual = self.oracle.weighted_prin_comp(self.reports)[1]
        result = []
        for i in range(len(actual)):
            result.append((actual[i] - expected[i])**2)
        self.assertLess(sum(result), 0.000000000001)

    def test_main(self):
        main()
        main(argv=('', '-h'))
        self.assertRaises(main(argv=('', '-q')))

    def tearDown(self):
        del self.reports
        del self.c
        del self.c2
        del self.c3

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestConsensus)
    unittest.TextTestRunner(verbosity=2).run(suite)
