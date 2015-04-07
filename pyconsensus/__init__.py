#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Consensus mechanism for Augur/Truthcoin.

pyconsensus is a Python implementation of the Augur/Truthcoin consensus
mechanism, described in detail at https://github.com/psztorc/Truthcoin.

Usage:

    from pyconsensus import Oracle

    # Example report matrix:
    #   - each row represents a reporter
    #   - each column represents an event in a prediction market
    my_reports = [[0.2, 0.7, -1, -1],
                  [0.3, 0.5, -1, -1],
                  [0.1, 0.7, -1, -1],
                  [0.5, 0.7,  1, -1],
                  [0.1, 0.2,  1,  1],
                  [0.1, 0.2,  1,  1]]
    my_event_bounds = [
        {"scaled": True, "min": 0.1, "max": 0.5},
        {"scaled": True, "min": 0.2, "max": 0.7},
        {"scaled": False, "min": -1, "max": 1},
        {"scaled": False, "min": -1, "max": 1},
    ]

    oracle = Oracle(reports=my_reports, event_bounds=my_event_bounds)
    oracle.consensus()

"""
from __future__ import division, absolute_import
import sys
import os
import getopt
import json
import warnings
from pprint import pprint
from copy import deepcopy
import numpy as np
import pandas as pd
from weightedstats import weighted_median
from six.moves import xrange as range

__title__      = "pyconsensus"
__version__    = "0.5.5"
__author__     = "Jack Peterson and Paul Sztorc"
__license__    = "GPL"
__maintainer__ = "Jack Peterson"
__email__      = "jack@tinybike.net"

pd.set_option("display.max_rows", 25)
pd.set_option("display.width", 1000)
np.set_printoptions(linewidth=225,
                    suppress=True,
                    formatter={"float": "{: 0.6f}".format})

class Oracle(object):

    def __init__(self, reports=None, event_bounds=None, reputation=None,
                 catch_tolerance=0.1, alpha=0.1, verbose=False,
                 aux=None, algorithm="fixed-variance", variance_threshold=0.9):
        """
        Args:
          reports (list-of-lists): reports matrix; rows = reporters, columns = Events.
          event_bounds (list): list of dicts for each Event
            {
              scaled (bool): True if scalar, False if binary (boolean)
              min (float): minimum allowed value (-1 if binary)
              max (float): maximum allowed value (1 if binary)
            }

        """
        self.NO = 1.0
        self.YES = 2.0
        self.BAD = 1.5
        self.NA = 0.0
        self.reports = np.ma.masked_array(reports, np.isnan(reports))
        self.num_reports = len(reports)
        self.num_events = len(reports[0])
        self.event_bounds = event_bounds
        self.catch_tolerance = catch_tolerance
        self.alpha = alpha  # reputation smoothing parameter
        self.verbose = verbose
        self.algorithm = algorithm
        self.variance_threshold = variance_threshold
        self.num_components = -1
        self.convergence = False
        self.aux = aux
        if reputation is None:
            self.weighted = False
            self.total_rep = self.num_reports
            self.reptokens = np.ones(self.num_reports).astype(int)
            self.reputation = np.array([1 / float(self.num_reports)] * self.num_reports)
        else:
            self.weighted = True
            self.total_rep = sum(np.array(reputation).ravel())
            self.reptokens = np.array(reputation).ravel().astype(int)
            self.reputation = np.array([i / float(self.total_rep) for i in reputation])

    def normalize(self, v):
        """Proportional distance from zero."""
        v = abs(v)
        if np.sum(v) == 0:
            v += 1
        return v / np.sum(v)

    def catch(self, X):
        """Forces continuous values into bins at NO, BAD, and YES."""
        if X < self.BAD - self.catch_tolerance:
            return self.NO
        elif X > self.BAD + self.catch_tolerance:
            return self.YES
        else:
            return self.BAD

    def interpolate(self, reports):
        """Uses existing data and reputations to fill missing observations.
        Weighted average/median using all available (non-nan) data.

        """
        # Rescale scaled events
        if self.event_bounds is not None:
            for i in range(self.num_events):
                if self.event_bounds[i]["scaled"]:
                    reports[:,i] = (reports[:,i] - self.event_bounds[i]["min"]) / float(self.event_bounds[i]["max"] - self.event_bounds[i]["min"])

        # Interpolation to fill the missing observations
        for j in range(self.num_events):
            if reports[:,j].mask.any():
                total_active_reputation = 0
                active_reputation = []
                active_reports = []
                nan_indices = []
                num_present = 0
                for i in range(self.num_reports):
                    if reports[i,j] != np.nan:
                        total_active_reputation += self.reputation[i]
                        active_reputation.append(self.reputation[i])
                        active_reports.append(reports[i,j])
                        num_present += 1
                    else:
                        nan_indices.append(i)
                if not self.event_bounds[j]["scaled"]:
                    guess = 0
                    for i in range(num_present):
                        active_reputation[i] /= total_active_reputation
                        guess += active_reputation[i] * active_reports[i]
                    guess = self.catch(guess)
                else:
                    for i in range(num_present):
                        active_reputation[i] /= total_active_reputation
                    guess = weighted_median(active_reports, weights=active_reputation)
                for nan_index in nan_indices:
                    reports[nan_index,j] = guess
        return reports

    def col_centering(self, reports_filled, standardize, bias):
        # Compute the weighted mean (of all reporters) for each event
        weighted_mean = np.dot(self.reputation, reports_filled)

        # Each report's difference from the mean of its event (column)
        cntr = np.matrix(reports_filled - weighted_mean)

        # standardize 
        if standardize:
            cntr /= np.std(cntr, axis=0, ddof=bias)

        # weight:
        # elementwise multiply each column of the difference-from-mean matrix
        # with the reputation vector
        return np.multiply(cntr, np.matrix(self.reputation).T)

    def collapse(self, reports_filled, order=4, standardize=True, axis=0, bias=0):
        """Internal sum over all tensor fibers except first"""
        # Center/whiten data
        cntr = self.col_centering(reports_filled, 
                                  standardize=standardize,
                                  bias=bias)

        # Vector collapse: internal sum across tensor fibers
        # import pdb; pdb.set_trace()
        rowsums = np.power(np.sum(cntr, 1).T, order - 1)
        return rowsums.dot(cntr) / (self.num_reports - bias)

    def wpca(self, reports_filled):
        # Compute the weighted mean (of all reporters) for each event
        weighted_mean = np.ma.average(reports_filled,
                                      axis=0,
                                      weights=self.reputation.tolist())

        # Each report's difference from the mean of its event (column)
        wcd = np.matrix(reports_filled - weighted_mean)
        tokens = [int(r * 1e6) for r in self.reputation]

        # Compute the unbiased weighted population covariance
        # (for uniform weights, equal to np.cov(reports_filled.T, bias=1))
        # wcd.T.dot(R).dot(wcd) / float(1 - np.sum(self.reputation**2))
        covariance_matrix = np.ma.multiply(wcd.T, tokens).dot(wcd) / float(np.sum(tokens) - 1)

        # H is the un-normalized eigenvector matrix
        try:
            H = np.linalg.svd(covariance_matrix)[0]
        except Exception as exc:
            print exc
            print
            print "Covariance matrix:"
            print covariance_matrix.data
            print
            print "Reports (filled):"
            print reports_filled.data
            print 
            print "Reputation:"
            print self.reputation
            print
            import pdb; pdb.set_trace()

        # Normalize loading by Euclidean distance
        first_loading = np.ma.masked_array(H[:,0] / np.sqrt(np.sum(H[:,0]**2)))
        first_score = np.dot(wcd, first_loading)

        # print "pyconsensus svd:"
        # print np.array(first_loading)
        # print np.array(np.ma.masked_array(H[:,1] / np.sqrt(np.sum(H[:,1]**2))))
        # print np.array(np.ma.masked_array(H[:,2] / np.sqrt(np.sum(H[:,2]**2))))
        # print np.array(np.ma.masked_array(H[:,3] / np.sqrt(np.sum(H[:,3]**2))))

        return weighted_mean, wcd, covariance_matrix, first_loading, first_score

    def lie_detector(self, reports_filled):
        """Calculates new reputations using a weighted Principal Component
        Analysis (PCA).

        Weights are the number of coins people start with, so the aim of this
        weighting is to count 1 report for each of their coins -- e.g., guy with 10
        coins effectively gets 10 reports, guy with 1 coin gets 1 report, etc.
        
        The reports matrix has reporters as rows and events as columns.

        """
        first_loading = np.ma.masked_array(np.zeros(self.num_events))
        first_score = np.ma.masked_array(np.zeros(self.num_reports))

        # Use the largest eigenvector only
        if self.algorithm == "sztorc":
            weighted_mean, wcd, covariance_matrix, first_loading, first_score = self.wpca(reports_filled)
            nc = self.nonconformity(first_score, reports_filled)

        elif self.algorithm == "big-five":
            weighted_mean, wcd, covariance_matrix, first_loading, first_score = self.wpca(reports_filled)
            U, Sigma, Vt = np.linalg.svd(covariance_matrix)
            variance_explained = np.cumsum(Sigma / np.trace(covariance_matrix))
            for i in range(5):
                loading = U.T[i]
                score = np.dot(wcd, loading)
                if i == 0:
                    net_score = Sigma[i] * score
                else:
                    net_score += Sigma[i] * score
            nc = self.nonconformity(net_score, reports_filled)

        elif self.algorithm == "first-third":
            weighted_mean, wcd, covariance_matrix, first_loading, first_score = self.wpca(reports_filled)
            U, Sigma, Vt = np.linalg.svd(covariance_matrix)
            variance_explained = np.cumsum(Sigma / np.trace(covariance_matrix))
            for i in range(int(np.ceil(self.num_events / 3.0))):
                loading = U.T[i]
                score = np.dot(wcd, loading)
                if i == 0:
                    net_score = Sigma[i] * score
                else:
                    net_score += Sigma[i] * score
            if self.aux is not None and "cokurt" in self.aux:
                net_score += self.aux["cokurt"]
            # print "net_score:"
            # print np.array(net_score)
            # print
            nc = self.nonconformity(net_score, reports_filled)

        # Fixed-variance threshold: eigenvalue-weighted sum of score vectors
        elif self.algorithm == "fixed-variance":
            weighted_mean, wcd, covariance_matrix, first_loading, first_score = self.wpca(reports_filled)
            U, Sigma, Vt = np.linalg.svd(covariance_matrix)
            # Sigma /= np.sum(Sigma)
            # print "U:"
            # print U
            # print "Sigma:"
            # print Sigma
            # print
            # print np.var(U)
            # print 
            # print "Total variance:"
            # print np.trace(covariance_matrix)
            # print
            variance_explained = np.cumsum(Sigma / np.trace(covariance_matrix))
            # import pdb; pdb.set_trace()
            # variance_explained = np.cumsum(Sigma)
            for i, var_exp in enumerate(variance_explained):
                loading = U.T[i]
                score = np.dot(wcd, loading)
                if i == 0:
                    net_score = Sigma[i] * score
                else:
                    net_score += Sigma[i] * score
                if var_exp > self.variance_threshold: break
            self.num_components = i
            nc = self.nonconformity(net_score, reports_filled)            

        # Sum over all events in the ballot; the ratio of this sum to
        # the total covariance (over all events, across all reporters)
        # is each reporter's contribution to the overall variability.
        elif self.algorithm == "covariance":
            nc = self.nonconformity(self.aux["cov"], reports_filled)

        # Sum over all events in the ballot; the ratio of this sum to
        # the total cokurtosis is that reporter's contribution.
        elif self.algorithm == "cokurtosis":
            if self.aux is not None and "cokurt" in self.aux:
                nc = self.nonconformity(self.aux["cokurt"], reports_filled)
            else:
                nc = np.array(self.collapse(reports_filled, axis=0)).ravel()

        elif self.algorithm == "virial":
            nc = self.nonconformity(self.aux["H"], reports_filled)

        # Use adjusted nonconformity scores to update Reputation fractions
        # import pdb; pdb.set_trace()
        this_rep = self.normalize(
            nc * (self.reputation / np.mean(self.reputation)).T
        )
        return {
            "first_loading": first_loading,
            "old_rep": self.reputation.T,
            "this_rep": this_rep,
            "smooth_rep": self.alpha*this_rep + (1-self.alpha)*self.reputation.T,
        }

    def nonconformity(self, scores, reports):
        """Adjusted nonconformity scores for Reputation redistribution"""
        set1 = scores + np.abs(np.min(scores))
        set2 = scores - np.max(scores)
        old = np.dot(self.reputation.T, reports)
        new1 = np.dot(self.normalize(set1), reports)
        new2 = np.dot(self.normalize(set2), reports)
        ref_ind = np.sum((new1 - old)**2) - np.sum((new2 - old)**2)
        self.convergence = True
        nc = set1 if ref_ind <= 0 else set2
        return nc

    def consensus(self):
        # Handle missing values
        reports_filled = self.interpolate(self.reports)

        # Consensus - Row Players
        player_info = self.lie_detector(reports_filled)

        # Column Players (The Event Creators)
        outcomes_raw = np.dot(player_info['smooth_rep'], reports_filled).squeeze()

        # Discriminate Based on Contract Type
        if self.event_bounds is not None:
            for i in range(reports_filled.shape[1]):

                # Our Current best-guess for this Scaled Event (weighted median)
                if self.event_bounds[i]["scaled"]:
                    outcomes_raw[i] = weighted_median(
                        reports_filled[:,i].data,
                        weights=player_info["smooth_rep"].ravel().data,
                    )

        # The Outcome (Discriminate Based on Contract Type)
        outcomes_adj = []
        for i, raw in enumerate(outcomes_raw):
            if self.event_bounds is not None and self.event_bounds[i]["scaled"]:
                outcomes_adj.append(raw)
            else:
                outcomes_adj.append(self.catch(raw))

        outcomes_final = []
        for i, raw in enumerate(outcomes_raw):
            outcomes_final.append(outcomes_adj[i])
            if self.event_bounds is not None and self.event_bounds[i]["scaled"]:
                outcomes_final[i] *= self.event_bounds[i]["max"] - self.event_bounds[i]["min"]
                outcomes_final[i] += self.event_bounds[i]["min"]

        certainty = []
        for i, adj in enumerate(outcomes_adj):
            certainty.append(sum(player_info["smooth_rep"][reports_filled[:,i] == adj]))

        certainty = np.array(certainty)
        consensus_reward = self.normalize(certainty)
        avg_certainty = np.mean(certainty)

        # Participation: information about missing values
        na_mat = self.reports * 0
        na_mat[na_mat.mask] = 1  # indicator matrix for missing

        # Participation Within Events (Columns)
        # % of reputation that answered each Event
        participation_columns = 1 - np.dot(player_info['smooth_rep'], na_mat)

        # Participation Within Agents (Rows)
        # Democracy Option - all Events treated equally.
        participation_rows = 1 - na_mat.sum(axis=1) / na_mat.shape[1]

        # General Participation
        percent_na = 1 - np.mean(participation_columns)

        # Possibly integrate two functions of participation? Chicken and egg problem...
        if self.verbose:
            print('*Participation Information*')
            print('Voter Turnout by question')
            print(participation_columns)
            print('Voter Turnout across questions')
            print(participation_rows)

        # Combine Information
        # Row
        na_bonus_reporters = self.normalize(participation_rows)
        reporter_bonus = na_bonus_reporters * percent_na + player_info['smooth_rep'] * (1 - percent_na)

        # Column
        na_bonus_events = self.normalize(participation_columns)
        author_bonus = na_bonus_events * percent_na + consensus_reward * (1 - percent_na)

        return {
            'original': self.reports.data,
            'filled': reports_filled.data,
            'agents': {
                'old_rep': player_info['old_rep'],
                'this_rep': player_info['this_rep'],
                'smooth_rep': player_info['smooth_rep'],
                'na_row': na_mat.sum(axis=1).data.tolist(),
                'participation_rows': participation_rows.data.tolist(),
                'relative_part': na_bonus_reporters.data.tolist(),
                'reporter_bonus': reporter_bonus.data.tolist(),
                },
            'events': {
                'adj_first_loadings': player_info['first_loading'].data.tolist(),
                'outcomes_raw': outcomes_raw.data.tolist(),
                'consensus_reward': consensus_reward,
                'certainty': certainty,
                'NAs Filled': na_mat.sum(axis=0).data.tolist(),
                'participation_columns': participation_columns.data.tolist(),
                'author_bonus': author_bonus.data.tolist(),
                'outcomes_adjusted': outcomes_adj,
                'outcomes_final': outcomes_final,
                },
            'participation': 1 - percent_na,
            'avg_certainty': avg_certainty,
            'convergence': self.convergence,
            'components': self.num_components,
        }

def main(argv=None):
    if argv is None:
        argv = sys.argv
    try:
        short_opts = 'hxmst'
        long_opts = ['help', 'example', 'missing', 'scaled', 'test']
        opts, vals = getopt.getopt(argv[1:], short_opts, long_opts)
    except getopt.GetoptError as e:
        sys.stderr.write(e.msg)
        sys.stderr.write("for help use --help")
        return 2
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(__doc__)
            return 0
        elif opt in ('-t', '--test'):
            # reports = np.array([[  0.837698,   0.49452,   2.54352 ],
            #                     [ -0.294096,  -0.39636,   0.728619],
            #                     [ -1.62089 ,  -0.44919,   1.20592 ],
            #                     [ -1.06458 ,  -0.68214,  -1.12841 ],
            #                     [  2.14341 ,   0.7309 ,   0.644968],
            #                     [ -0.284139,  -1.133  ,   1.98615 ],
            #                     [  1.19879 ,   2.55633,  -0.526461],
            #                     [ -0.032277,   0.11701,  -0.249265],
            #                     [ -1.02516 ,  -0.44665,   2.50556 ],
            #                     [ -0.515272,  -0.578  ,   0.515139],
            #                     [  0.259474,  -1.24193,   0.105051],
            #                     [  0.178546,  -0.80547,  -0.016838],
            #                     [ -0.607696,  -0.21319,  -1.40657 ],
            #                     [  0.372248,   0.93341,  -0.667086],
            #                     [ -0.099814,   0.52698,  -0.253867],
            #                     [  0.743166,  -0.79375,   2.11131 ],
            #                     [  0.109262,  -1.28021,  -0.415184],
            #                     [  0.499346,  -0.95897,  -2.24336 ],
            #                     [ -0.191825,  -0.59756,  -0.63292 ],
            #                     [ -1.98255 ,  -1.5936 ,  -0.935766],
            #                     [ -0.317612,   1.33143,  -0.46866 ],
            #                     [  0.666652,  -0.81507,   0.370959],
            #                     [ -0.761136,   0.10966,  -0.997161],
            #                     [ -1.09972 ,   0.28247,  -0.846566]])
            # # reputation = [0.4, 1.0, 0.6]
            # reputation = [ 91 ,
            #                65 ,
            #                57 ,
            #                34 ,
            #                83 ,
            #                70 ,
            #                74 ,
            #                26 ,
            #                37 ,
            #                21 ,
            #                62 ,
            #                22 ,
            #                91 ,
            #                83 ,
            #                89 ,
            #                39 ,
            #                20 ,
            #                83 ,
            #                36 ,
            #                75 ,
            #                43 ,
            #                78 ,
            #                83 ,
            #                17 ]
            reputation = [
                0.031033099688052984,
                0.06603411776796245,
                0.06603411776796245,
                0.02966635564758383,
                0.06603411776796245,
                0.029557191000632158,
                0.0281089563813696,
                0.031817973430730624,
                0.03186036639673597,
                0.06603411776796245,
                0.02309958697586698,
                0.025229389348775236,
                0.0340521799192279,
                0.03512430867041376,
                0.06603411776796245,
                0.024612072665886477,
                0.030564305942646947,
                0.031504825512057105,
                0.027273890173105043,
                0.06603411776796245,
                0.03253880677695827,
                0.03127655854061946,
                0.027577138895966653,
                0.06603411776796245,
                0.03286416965763393,
            ]
            reports = np.array([[  1.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  1.0, -1.0,  1.0 ],
                                [ -1.0,  1.0,  1.0,  0.0, -1.0, -1.0, -1.0,  0.0, -1.0,  0.0 ],
                                [ -1.0,  1.0,  1.0,  0.0, -1.0, -1.0, -1.0,  0.0, -1.0,  0.0 ],
                                [  0.0, -1.0,  0.0,  0.0,  0.0, -1.0, -1.0, -1.0,  1.0,  1.0 ],
                                [ -1.0,  1.0,  1.0,  0.0, -1.0, -1.0, -1.0,  0.0, -1.0,  0.0 ],
                                [ -1.0,  0.0,  0.0, -1.0,  0.0,  0.0,  1.0, -1.0,  0.0,  0.0 ],
                                [  0.0,  1.0,  0.0,  1.0,  0.0,  0.0,  1.0,  1.0,  0.0, -1.0 ],
                                [  0.0,  0.0, -1.0, -1.0,  0.0,  1.0, -1.0,  1.0,  1.0, -1.0 ],
                                [  0.0,  0.0,  0.0,  1.0,  0.0,  1.0, -1.0,  1.0,  1.0,  0.0 ],
                                [ -1.0,  1.0,  1.0,  0.0, -1.0, -1.0, -1.0,  0.0, -1.0,  0.0 ],
                                [ -1.0,  1.0,  0.0,  0.0, -1.0,  1.0,  1.0,  1.0, -1.0,  0.0 ],
                                [  1.0, -1.0, -1.0, -1.0,  0.0,  0.0,  1.0, -1.0,  1.0, -1.0 ],
                                [  1.0, -1.0,  1.0, -1.0,  1.0,  1.0, -1.0,  1.0,  1.0,  1.0 ],
                                [  0.0,  1.0,  0.0, -1.0,  0.0,  1.0, -1.0,  0.0, -1.0,  1.0 ],
                                [ -1.0,  1.0,  1.0,  0.0, -1.0, -1.0, -1.0,  0.0, -1.0,  0.0 ],
                                [  0.0,  0.0,  1.0, -1.0,  0.0,  0.0,  0.0, -1.0,  0.0,  1.0 ],
                                [ -1.0, -1.0,  1.0, -1.0, -1.0,  1.0,  0.0, -1.0,  1.0,  0.0 ],
                                [ -1.0,  1.0, -1.0,  1.0,  0.0, -1.0, -1.0,  1.0,  1.0,  1.0 ],
                                [ -1.0, -1.0,  0.0, -1.0,  1.0,  0.0, -1.0,  0.0, -1.0,  0.0 ],
                                [ -1.0,  1.0,  1.0,  0.0, -1.0, -1.0, -1.0,  0.0, -1.0,  0.0 ],
                                [ -1.0, -1.0, -1.0,  1.0, -1.0,  1.0,  0.0,  1.0, -1.0,  1.0 ],
                                [  1.0,  1.0,  0.0,  1.0,  1.0,  0.0, -1.0,  1.0,  1.0,  1.0 ],
                                [ -1.0,  1.0,  1.0,  1.0, -1.0,  0.0,  1.0,  0.0, -1.0, -1.0 ],
                                [ -1.0,  1.0,  1.0,  0.0, -1.0, -1.0, -1.0,  0.0, -1.0,  0.0 ],
                                [ -1.0,  1.0,  1.0,  1.0, -1.0,  0.0,  1.0,  0.0, -1.0, -1.0 ]])
            aux = {
                "cokurt": [
                    -0.00611437 ,
                     0.133337   ,
                     0.133337   ,
                    -0.00933107 ,
                     0.133337   ,
                     0.00332694 ,
                     0.00715548 ,
                    -0.0096433  ,
                    -0.00320225 ,
                     0.133337   ,
                     0.0167826  ,
                    -0.0315066  ,
                    -0.0101499  ,
                     0.040571   ,
                     0.133337   ,
                     0.0156032  ,
                     0.004482   ,
                     0.00613153 ,
                    -0.0010924  ,
                     0.133337   ,
                    -0.0127911  ,
                     0.00157463 ,
                     0.0274216  ,
                     0.133337   ,
                     0.0274216  ,
                ],
            }
            oracle = Oracle(reports=reports,
                            reputation=reputation,
                            algorithm="cokurtosis",
                            aux=aux)
            A = oracle.consensus()
            # print(pd.DataFrame(A["events"]))
            print(pd.DataFrame(A["agents"]))
        elif opt in ('-x', '--example'):
            # # new: true=1, false=-1, indeterminate=0.5, no response=0
            reports = np.array([[  1,  1, -1, -1],
                                [  1, -1, -1, -1],
                                [  1,  1, -1, -1],
                                [  1,  1,  1, -1],
                                [ -1, -1,  1,  1],
                                [ -1, -1,  1,  1]])
            reputation = [2, 10, 4, 2, 7, 1]
            oracle = Oracle(reports=reports,
                            reputation=reputation,
                            algorithm="sztorc")
            # oracle = Oracle(reports=reports, run_ica=True)
            A = oracle.consensus()
            print(pd.DataFrame(A["events"]))
            print(pd.DataFrame(A["agents"]))
        elif opt in ('-m', '--missing'):
            # true=1, false=-1, indeterminate=0.5, no response=np.nan
            reports = np.array([[      1,  1, -1, np.nan],
                                [      1, -1, -1,     -1],
                                [      1,  1, -1,     -1],
                                [      1,  1,  1,     -1],
                                [ np.nan, -1,  1,      1],
                                [     -1, -1,  1,      1]])
            reputation = [2, 10, 4, 2, 7, 1]
            oracle = Oracle(reports=reports,
                            reputation=reputation,
                            algorithm="sztorc")
            A = oracle.consensus()
            print(pd.DataFrame(A["events"]))
            print(pd.DataFrame(A["agents"]))
        elif opt in ('-s', '--scaled'):
            reports = np.array([[ 1,  1, -1, -1, 233, 16027.59],
                                [ 1, -1, -1, -1, 199,   np.nan],
                                [ 1,  1, -1, -1, 233, 16027.59],
                                [ 1,  1,  1, -1, 250,   np.nan],
                                [-1, -1,  1,  1, 435,  8001.00],
                                [-1, -1,  1,  1, 435, 19999.00]])
            event_bounds = [
                { "scaled": False, "min": -1,   "max": 1 },
                { "scaled": False, "min": -1,   "max": 1 },
                { "scaled": False, "min": -1,   "max": 1 },
                { "scaled": False, "min": -1,   "max": 1 },
                { "scaled": True,  "min":  0,   "max": 435 },
                { "scaled": True,  "min": 8000, "max": 20000 },
            ]
            oracle = Oracle(reports=reports, event_bounds=event_bounds)
            A = oracle.consensus()
            print(pd.DataFrame(A["events"]))
            print(pd.DataFrame(A["agents"]))

if __name__ == '__main__':
    sys.exit(main(sys.argv))
