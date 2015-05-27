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
from collections import Counter
from pprint import pprint
from copy import deepcopy
import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt
from scipy import cluster
from weightedstats import weighted_median
from six.moves import xrange as range

__title__      = "pyconsensus"
__version__    = "0.5.7"
__author__     = "Jack Peterson and Paul Sztorc"
__license__    = "GPL"
__maintainer__ = "Jack Peterson"
__email__      = "jack@tinybike.net"

pd.set_option("display.max_rows", 25)
pd.set_option("display.width", 1000)
np.set_printoptions(linewidth=225,
                    suppress=True,
                    formatter={"float": "{: 0.6f}".format})
NO = 1.0
YES = 2.0
BAD = 1.5
NA = 0.0

class Oracle(object):

    def __init__(self, reports=None, event_bounds=None, reputation=None,
                 catch_tolerance=0.1, alpha=0.1, verbose=False,
                 aux=None, algorithm="fixed-variance", variance_threshold=0.9,
                 max_components=5, hierarchy_threshold=0.5):
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
        self.hierarchy_threshold = hierarchy_threshold
        self.convergence = False
        self.aux = aux
        if self.num_events >= max_components:
            self.max_components = max_components
        else:
            self.max_components = self.num_events
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

    def wpca(self, reports_filled):
        # Compute the weighted mean (of all reporters) for each event
        weighted_mean = np.ma.average(reports_filled,
                                      axis=0,
                                      weights=self.reputation.tolist())

        # Each report's difference from the mean of its event (column)
        wcd = np.matrix(reports_filled - weighted_mean)
        tokens = [int(r * 1e6) for r in self.reputation]

        # Compute the unbiased weighted population covariance
        covariance_matrix = np.ma.multiply(wcd.T, tokens).dot(wcd) / float(np.sum(tokens) - 1)

        # H is the un-normalized eigenvector matrix
        try:
            H = np.linalg.svd(covariance_matrix)[0]
        except Exception as exc:
            print exc

        # Normalize loading by Euclidean distance
        first_loading = np.ma.masked_array(H[:,0] / np.sqrt(np.sum(H[:,0]**2)))
        first_score = np.dot(wcd, first_loading)

        return weighted_mean, wcd, covariance_matrix, first_loading, first_score

    def lie_detector(self, reports_filled):
        """Weights are the number of coins people start with, so the aim of this
        weighting is to count 1 report for each of their coins -- e.g., guy with 10
        coins effectively gets 10 reports, guy with 1 coin gets 1 report, etc.
        
        The reports matrix has reporters as rows and events as columns.

        """
        first_loading = np.ma.masked_array(np.zeros(self.num_events))
        first_score = np.ma.masked_array(np.zeros(self.num_reports))
        scores = np.zeros(self.num_reports)
        nc = np.zeros(self.num_reports)

        if self.verbose:
            print "pyconsensus [%s]:\n" % self.algorithm

        # Use the largest eigenvector only
        if self.algorithm == "PCA":
            weighted_mean, wcd, covariance_matrix, first_loading, first_score = self.wpca(reports_filled)
            nc = self.nonconformity(first_score, reports_filled)
            scores = first_score

        elif self.algorithm == "big-five":
            weighted_mean, wcd, covariance_matrix, first_loading, first_score = self.wpca(reports_filled)
            U, Sigma, Vt = np.linalg.svd(covariance_matrix)
            net_score = np.zeros(self.num_reports)
            for i in range(self.max_components):
                loading = U.T[i]
                if loading[0] < 0:
                    loading *= -1
                score = Sigma[i] * wcd.dot(loading)
                net_score += score
                if self.verbose:
                    print "  Eigenvector %d:" % i, np.round(loading, 6)
                    print "  Eigenvalue %d: " % i, Sigma[i]
                    print "  Projection:    ", np.round(score, 6)
                    print "  Nonconformity:", np.round(net_score, 6)
                    print
            nc = self.nonconformity(net_score, reports_filled)
            scores = net_score

        elif self.algorithm == "k-means":
            weighted_mean, wcd, covariance_matrix, first_loading, first_score = self.wpca(reports_filled)
            reports = cluster.vq.whiten(wcd)
            num_clusters = int(np.ceil(np.sqrt(len(reports))))
            # num_clusters = 4
            centroids,_ = cluster.vq.kmeans(reports, num_clusters)
            clustered,_ = cluster.vq.vq(reports, centroids)
            counts = Counter(list(clustered)).most_common()
            new_rep = {}
            for i, c in enumerate(counts):
                new_rep[c[0]] = c[1]
            new_rep_list = []
            for c in clustered:
                new_rep_list.append(new_rep[c])
            new_rep_list = np.array(new_rep_list) - min(new_rep_list)
            nc = new_rep_list / sum(new_rep_list)

        elif self.algorithm == "hierarchy":
            weighted_mean, wcd, covariance_matrix, first_loading, first_score = self.wpca(reports_filled)
            clustered = cluster.hierarchy.fclusterdata(wcd, self.hierarchy_threshold, criterion='distance')
            counts = Counter(list(clustered)).most_common()
            new_rep = {}
            for i, c in enumerate(counts):
                new_rep[c[0]] = c[1]
            new_rep_list = []
            for c in clustered:
                new_rep_list.append(new_rep[c])
            new_rep_list = np.array(new_rep_list) - min(new_rep_list)
            nc = new_rep_list / sum(new_rep_list)

        elif self.algorithm == "absolute":
            weighted_mean, wcd, covariance_matrix, first_loading, first_score = self.wpca(reports_filled)
            mean = np.mean(first_score)
            distance = np.abs((first_score - mean))
            nc = 1 / np.square(1 + distance)
            scores = first_score

        # Fixed-variance threshold: eigenvalue-weighted sum of score vectors
        elif self.algorithm == "fixed-variance":
            weighted_mean, wcd, covariance_matrix, first_loading, first_score = self.wpca(reports_filled)
            U, Sigma, Vt = np.linalg.svd(covariance_matrix)
            variance_explained = np.cumsum(Sigma / np.trace(covariance_matrix))
            net_score = np.zeros(self.num_reports)
            negative = False
            for i, var_exp in enumerate(variance_explained):
                loading = U.T[i]
                if loading[0] < 0:
                    loading *= -1
                score = Sigma[i] * wcd.dot(loading)
                net_score += score
                if self.verbose:
                    print "  Eigenvector %d:" % i, np.round(loading, 6)
                    print "  Eigenvalue %d: " % i, Sigma[i], "(%s%% variance explained)" % np.round(var_exp * 100, 3)
                    print "  Projection:    ", np.round(score, 6)
                    print "  Nonconformity:", np.round(net_score, 6)
                    print "  Variance explained:", var_exp, i
                    print
                if var_exp >= self.variance_threshold: break
            self.num_components = i + 1
            nc = self.nonconformity(net_score, reports_filled)
            scores = net_score

        # Sum over all events in the ballot; the ratio of this sum to
        # the total cokurtosis is that reporter's contribution.
        elif self.algorithm == "cokurtosis":
            nc = self.nonconformity(self.aux["cokurt"], reports_filled)
            scores = self.aux["cokurt"]

        # Use adjusted nonconformity scores to update Reputation fractions
        this_rep = self.normalize(
            nc * (self.reputation / np.mean(self.reputation)).T
        )
        if self.verbose:
            print "  Adjusted:  ", nc
            print "  Reputation:", this_rep
            print
        return {
            "first_loading": first_loading,
            "scores": scores,
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
        outcomes_raw = np.dot(player_info['smooth_rep'], reports_filled)
        if outcomes_raw.shape != (1,):
            outcomes_raw = outcomes_raw.squeeze()

        # Discriminate Based on Contract Type
        if self.event_bounds is not None:
            for i in range(reports_filled.shape[1]):

                # Our Current best-guess for this Scaled Event (weighted median)
                if self.event_bounds[i]["scaled"]:
                    outcomes_raw[i] = weighted_median(
                        reports_filled[:,i],
                        weights=player_info["smooth_rep"].ravel(),
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
                'scores': player_info['scores'],
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
        short_opts = 'hxmst:'
        long_opts = ['help', 'example', 'missing', 'scaled', 'test=']
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
            testalgo = "hierarchy"
            if arg == "1":
                reports = np.array([[ YES, YES,  NO,  NO ],
                                    [ YES,  NO,  NO,  NO ],
                                    [ YES, YES,  NO,  NO ],
                                    [ YES, YES, YES,  NO ],
                                    [  NO,  NO, YES, YES ],
                                    [  NO,  NO, YES, YES ]])
            elif arg == "2":
                reports = np.array([[ YES, YES,  NO,  NO ],
                                    [ YES, YES,  NO,  NO ],
                                    [ YES, YES,  NO,  NO ],
                                    [ YES, YES,  NO,  NO ],
                                    [ YES, YES,  NO,  NO ],
                                    [ YES, YES,  NO,  NO ],
                                    [ YES, YES, YES,  NO ],
                                    [ YES, YES, YES,  NO ],
                                    [ YES, YES, YES,  NO ],
                                    [ YES, YES, YES,  NO ],
                                    [ YES, YES, YES,  NO ]])
            elif arg == "3":
                reports =  np.array([[ YES,  YES,   NO,  NO,  YES,  YES,  NO,   NO,  YES,  YES,   NO,   NO,  YES],
                                     [ YES,  YES,   NO,  NO,  YES,  YES,  NO,   NO,  YES,  YES,   NO,   NO,  YES],
                                     [ YES,  YES,   NO,  NO,  YES,  YES,  NO,   NO,  YES,  YES,   NO,   NO,  YES],
                                     [ YES,  YES,   NO,  NO,  YES,  YES,  NO,   NO,  YES,  YES,   NO,   NO,  YES],
                                     [ YES,  YES,   NO,  NO,  YES,  YES,  NO,   NO,  YES,  YES,   NO,   NO,  YES],
                                     [ YES,  YES,   NO,  NO,  YES,  YES,  NO,   NO,  YES,  YES,   NO,   NO,  YES],

                                     [  NO,   NO,   NO, YES,   NO,   NO,  NO,  YES,   NO,   NO,   NO,  YES,   NO],
                                     
                                     [ YES,  YES,  YES,  NO,  YES,  YES,  YES,  NO,  YES,  YES,  YES,   NO,  YES],
                                     [ YES,  YES,  YES,  NO,  YES,  YES,  YES,  NO,  YES,  YES,  YES,   NO,  YES],
                                     [ YES,  YES,  YES,  NO,  YES,  YES,  YES,  NO,  YES,  YES,  YES,   NO,  YES],
                                     [ YES,  YES,  YES,  NO,  YES,  YES,  YES,  NO,  YES,  YES,  YES,   NO,  YES]])
            elif arg == "4":
                reports =  np.array([[ YES,  YES,   NO,   NO,  YES ],
                                     [ YES,  YES,   NO,   NO,  YES ],
                                     [ YES,  YES,   NO,   NO,  YES ],
                                     [ YES,  YES,   NO,   NO,  YES ],
                                     [ YES,  YES,   NO,   NO,  YES ],
                                     [ YES,  YES,   NO,   NO,  YES ],
                                     [ YES,  YES,   NO,   NO,  YES ],
                                     [ YES,  YES,   NO,   NO,  YES ],
                                     [ YES,  YES,   NO,   NO,  YES ],
                                     [ YES,  YES,   NO,   NO,  YES ],
                                     [ YES,  YES,   NO,   NO,  YES ],
                                     [ YES,  YES,   NO,   NO,  YES ],
                                     [ YES,  YES,   NO,   NO,  YES ],
                                     [ YES,  YES,   NO,   NO,  YES ],
                                     [ YES,  YES,   NO,   NO,  YES ],
                                     [  NO,   NO,   NO,  YES,   NO ],
                                     [ YES,  YES,  YES,   NO,  YES ],
                                     [ YES,  YES,  YES,   NO,  YES ],
                                     [ YES,  YES,  YES,   NO,  YES ],
                                     [ YES,  YES,  YES,   NO,  YES ],
                                     [ YES,  YES,  YES,   NO,  YES ],
                                     [ YES,  YES,   NO,   NO,  YES ],
                                     [ YES,  YES,   NO,   NO,  YES ],
                                     [ YES,  YES,   NO,   NO,  YES ],
                                     [ YES,  YES,   NO,   NO,  YES ]])

            elif arg == "5":
                reports = np.array([[ BAD,  NO,  NO, YES,  NO,  NO, YES, YES, BAD, BAD ],
                                    [ BAD, BAD,  NO, BAD, BAD, YES, YES, BAD, YES, BAD ],
                                    [  NO, YES, BAD, BAD,  NO, YES,  NO,  NO, BAD, BAD ],
                                    [ BAD, BAD, BAD, BAD, BAD,  NO,  NO,  NO, BAD, YES ],
                                    [  NO, YES, YES, BAD, BAD, YES, BAD, YES, BAD, YES ],
                                    [  NO, YES, YES, YES,  NO, BAD,  NO, BAD, BAD, BAD ],
                                    [  NO,  NO,  NO, YES,  NO,  NO,  NO, YES, BAD, YES ],
                                    [ BAD, BAD, BAD, YES, BAD, YES, BAD, BAD, YES,  NO ],
                                    [ BAD, BAD, BAD,  NO, BAD, YES, YES,  NO,  NO, BAD ],
                                    [ BAD, YES, BAD, YES,  NO,  NO, YES, YES,  NO, BAD ],
                                    [ YES, YES, BAD, BAD, BAD, YES, BAD, BAD, YES, YES ],
                                    [ YES, BAD, YES,  NO, YES, BAD, YES,  NO, YES, BAD ],
                                    [  NO,  NO,  NO, YES, YES, YES, BAD, YES, BAD,  NO ],
                                    [  NO,  NO,  NO, YES, YES, YES, BAD, YES, BAD,  NO ],
                                    [  NO,  NO,  NO, YES, YES, YES, BAD, YES, BAD,  NO ],
                                    [  NO,  NO,  NO, YES, YES, YES, BAD, YES, BAD,  NO ],
                                    [  NO,  NO,  NO, YES, YES, YES, BAD, YES, BAD,  NO ],
                                    [  NO,  NO,  NO, YES, YES, YES, BAD, YES, BAD,  NO ],
                                    [  NO,  NO,  NO, YES, YES, YES, BAD, YES, BAD,  NO ],
                                    [ BAD, BAD, BAD, YES, BAD, YES, BAD, BAD, YES,  NO ]])
            elif arg == "6":
                reports = np.array([[  NO,   NO,  YES,  YES,   NO,  YES,   NO,   NO,   NO,   NO ],
                                    [ YES,  YES,   NO,   NO,   NO,  YES,  YES,  YES,   NO,  YES ],
                                    [ YES,  YES,   NO,  YES,   NO,  YES,  YES,   NO,  YES,  YES ],
                                    [  NO,  YES,   NO,   NO,  YES,   NO,  YES,   NO,   NO,  YES ],
                                    [  NO,   NO,  YES,   NO,  YES,   NO,   NO,   NO,   NO,   NO ],
                                    [  NO,  YES,   NO,   NO,   NO,  YES,  YES,   NO,  YES,  YES ],
                                    [ YES,   NO,   NO,  YES,  YES,   NO,  YES,   NO,   NO,   NO ],
                                    [ YES,  YES,   NO,   NO,  YES,   NO,  YES,  YES,  YES,   NO ],
                                    [ YES,   NO,   NO,  YES,   NO,  YES,   NO,   NO,   NO,  YES ],
                                    [ YES,   NO,   NO,  YES,   NO,  YES,   NO,   NO,   NO,  YES ],
                                    [ YES,   NO,   NO,  YES,   NO,  YES,   NO,   NO,   NO,  YES ],
                                    [ YES,   NO,   NO,  YES,   NO,  YES,   NO,   NO,   NO,  YES ],
                                    [ YES,   NO,   NO,  YES,   NO,  YES,   NO,   NO,   NO,  YES ],
                                    [ YES,   NO,   NO,  YES,   NO,  YES,   NO,   NO,   NO,  YES ],
                                    [ YES,   NO,   NO,  YES,   NO,  YES,   NO,   NO,   NO,  YES ],
                                    [ YES,   NO,   NO,  YES,   NO,  YES,   NO,   NO,   NO,  YES ],
                                    [ YES,   NO,   NO,  YES,   NO,  YES,   NO,   NO,   NO,  YES ],
                                    [ YES,   NO,   NO,  YES,   NO,  YES,   NO,   NO,   NO,  YES ],
                                    [ YES,   NO,   NO,  YES,   NO,  YES,   NO,   NO,   NO,  YES ],
                                    [  NO,  YES,   NO,   NO,  YES,   NO,  YES,   NO,   NO,  YES ]])
            oracle = Oracle(reports=reports, algorithm=testalgo)
            A = oracle.consensus()
            print(reports)
            # print(pd.DataFrame(A["events"]))
            print(pd.DataFrame(A["agents"]))

        elif opt in ('-x', '--example'):
            reports = np.array([[ YES, YES,  NO,  NO],
                                [ YES,  NO,  NO,  NO],
                                [ YES, YES,  NO,  NO],
                                [ YES, YES, YES,  NO],
                                [  NO,  NO, YES, YES],
                                [  NO,  NO, YES, YES]])
            reputation = [2, 10, 4, 2, 7, 1]
            oracle = Oracle(reports=reports,
                            reputation=reputation,
                            algorithm="absolute")
            A = oracle.consensus()
            print(pd.DataFrame(A["events"]))
            print(pd.DataFrame(A["agents"]))
        elif opt in ('-m', '--missing'):
            reports = np.array([[    YES, YES,  NO, np.nan],
                                [    YES,  NO,  NO,     NO],
                                [    YES, YES,  NO,     NO],
                                [    YES, YES, YES,     NO],
                                [ np.nan,  NO, YES,    YES],
                                [     NO,  NO, YES,    YES]])
            reputation = [2, 10, 4, 2, 7, 1]
            oracle = Oracle(reports=reports,
                            reputation=reputation,
                            algorithm="PCA")
            A = oracle.consensus()
            print(pd.DataFrame(A["events"]))
            print(pd.DataFrame(A["agents"]))
        elif opt in ('-s', '--scaled'):
            reports = np.array([[ YES, YES,  NO,  NO, 233, 16027.59],
                                [ YES,  NO,  NO,  NO, 199,   np.nan],
                                [ YES, YES,  NO,  NO, 233, 16027.59],
                                [ YES, YES, YES,  NO, 250,   np.nan],
                                [  NO,  NO, YES, YES, 435,  8001.00],
                                [  NO,  NO, YES, YES, 435, 19999.00]])
            event_bounds = [
                { "scaled": False, "min": NO,   "max": 1 },
                { "scaled": False, "min": NO,   "max": 1 },
                { "scaled": False, "min": NO,   "max": 1 },
                { "scaled": False, "min": NO,   "max": 1 },
                { "scaled": True,  "min":  0,   "max": 435 },
                { "scaled": True,  "min": 8000, "max": 20000 },
            ]
            oracle = Oracle(reports=reports, event_bounds=event_bounds)
            A = oracle.consensus()
            print(pd.DataFrame(A["events"]))
            print(pd.DataFrame(A["agents"]))

if __name__ == '__main__':
    sys.exit(main(sys.argv))
