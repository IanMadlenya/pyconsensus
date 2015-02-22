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
from sklearn.decomposition import FastICA, PCA
from weightedstats import weighted_median
from six.moves import xrange as range

__title__      = "pyconsensus"
__version__    = "0.4.1"
__author__     = "Paul Sztorc and Jack Peterson"
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
                 catch_tolerance=0.1, alpha=0.1, verbose=False, aux=None,
                 algorithm="sztorc", variance_threshold=0.9):
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
        self.reports = np.ma.masked_array(reports, np.isnan(reports))
        self.num_reporters = len(reports)
        self.num_events = len(reports[0])
        self.event_bounds = event_bounds
        self.catch_tolerance = catch_tolerance
        self.alpha = alpha
        self.verbose = verbose
        self.algorithm = algorithm
        self.variance_threshold = variance_threshold
        self.num_components = -1
        self.aux = aux
        if reputation is None:
            self.weighted = False
            self.total_rep = self.num_reporters
            self.reptokens = np.ones(self.num_reporters).astype(int)
            self.reputation = np.array([1 / float(self.num_reporters)] * self.num_reporters)
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
        """Forces continuous values into bins at -1, 0, and 1."""
        center = 0
        if X < center - self.catch_tolerance:
            return -1
        elif X > center + self.catch_tolerance:
            return 1
        else:
            return 0

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
                for i in range(self.num_reporters):
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

    def weighted_pca(self, reports_filled):
        """Calculates new reputations using a weighted Principal Component
        Analysis (PCA).

        Weights are the number of coins people start with, so the aim of this
        weighting is to count 1 report for each of their coins -- e.g., guy with 10
        coins effectively gets 10 reports, guy with 1 coin gets 1 report, etc.
        
        The reports matrix has reporters as rows and events as columns.

        """
        convergence = False
        net_adj_prin_comp = None

        # Compute the weighted mean (of all reporters) for each event
        weighted_mean = np.ma.average(reports_filled,
                                      axis=0,
                                      weights=self.reputation.tolist())

        # Each report's difference from the mean of its event (column)
        mean_deviation = np.matrix(reports_filled - weighted_mean)

        # Compute the unbiased weighted population covariance
        # (for uniform weights, equal to np.cov(reports_filled.T, bias=1))
        covariance_matrix = np.ma.multiply(mean_deviation.T, self.reputation).dot(mean_deviation) / float(1 - np.sum(self.reputation**2))

        # H is the un-normalized eigenvector matrix
        H = np.linalg.svd(covariance_matrix)[0]

        # Normalize loading by Euclidean distance
        first_loading = np.ma.masked_array(H[:,0] / np.sqrt(np.sum(H[:,0]**2)))
        first_score = np.dot(mean_deviation, first_loading)

        if self.algorithm == "sztorc":

            set1 = first_score + np.abs(np.min(first_score))
            set2 = first_score - np.max(first_score)
            old = np.dot(self.reputation.T, reports_filled)
            new1 = np.dot(self.normalize(set1), reports_filled)
            new2 = np.dot(self.normalize(set2), reports_filled)
            ref_ind = np.sum((new1 - old)**2) - np.sum((new2 - old)**2)
            adj_prin_comp = set1 if ref_ind <= 0 else set2
            net_adj_prin_comp = adj_prin_comp
            convergence = True

        elif self.algorithm == "fixed-var-length":

            U, Sigma, Vt = np.linalg.svd(covariance_matrix)
            variance_explained = np.cumsum(Sigma / np.trace(covariance_matrix))
            length = 0
            for i, var_exp in enumerate(variance_explained):
                loading = U.T[i]
                score = Sigma[i] * np.dot(mean_deviation, loading)
                length += score**2
                if var_exp > self.variance_threshold: break
            if self.verbose:
                print i, "components"
            self.num_components = i
            length = np.sqrt(length)
            set1 = length + np.abs(np.min(length))
            set2 = length - np.max(length)
            old = np.dot(self.reputation.T, reports_filled)
            new1 = np.dot(self.normalize(set1), reports_filled)
            new2 = np.dot(self.normalize(set2), reports_filled)
            ref_ind = np.sum((new1 - old)**2) - np.sum((new2 - old)**2)
            net_adj_prin_comp = set1 if ref_ind <= 0 else set2
            # net_adj_prin_comp = 1 / np.abs(length)
            # net_adj_prin_comp /= np.sum(net_adj_prin_comp)
            convergence = True

        elif self.algorithm == "fixed-variance":

            U, Sigma, Vt = np.linalg.svd(covariance_matrix)
            variance_explained = np.cumsum(Sigma / np.trace(covariance_matrix))
            for i, var_exp in enumerate(variance_explained):
                loading = U.T[i]
                score = np.dot(mean_deviation, loading)
                if i == 0:
                    net_score = Sigma[i] * score
                else:
                    net_score += Sigma[i] * score
                if var_exp > self.variance_threshold: break
            if self.verbose:
                print i, "components"
            self.num_components = i
            set1 = net_score + np.abs(np.min(net_score))
            set2 = net_score - np.max(net_score)
            old = np.dot(self.reputation.T, reports_filled)
            new1 = np.dot(self.normalize(set1), reports_filled)
            new2 = np.dot(self.normalize(set2), reports_filled)
            ref_ind = np.sum((new1 - old)**2) - np.sum((new2 - old)**2)
            net_adj_prin_comp = set1 if ref_ind <= 0 else set2
            convergence = True            

        elif self.algorithm == "inverse-scores":

            # principal_components = PCA().fit_transform(covariance_matrix)
            principal_components = np.linalg.svd(covariance_matrix)[0]
            first_loading = principal_components[:,0]
            first_loading = np.ma.masked_array(first_loading / np.sqrt(np.sum(first_loading**2)))
            first_score = np.dot(mean_deviation, first_loading)

            # Normalized absolute inverse scores
            net_adj_prin_comp = 1 / np.abs(first_score)
            net_adj_prin_comp /= np.sum(net_adj_prin_comp)
            convergence = True

        elif self.algorithm == "ica-adjusted":
            ica = FastICA(n_components=self.num_events, whiten=False)
                          # random_state=0,
                          # max_iter=1000)
            while not convergence:
                with warnings.catch_warnings(record=True) as w:
                    try:
                        S_ = ica.fit_transform(covariance_matrix)   # Reconstruct signals
                        if len(w):
                            continue
                        else:
                            if self.verbose:
                                print "ICA loadings:"
                                print S_
                            S_first_loading = S_[:,0]
                            S_first_loading /= np.sqrt(np.sum(S_first_loading**2))
                            S_first_score = np.array(np.dot(mean_deviation, S_first_loading)).ravel()

                            S_set1 = S_first_score + np.abs(np.min(S_first_score))
                            S_set2 = S_first_score - np.max(S_first_score)
                            S_old = np.dot(self.reputation.T, reports_filled)
                            S_new1 = np.dot(self.normalize(S_set1), reports_filled)
                            S_new2 = np.dot(self.normalize(S_set2), reports_filled)
                            S_ref_ind = np.sum((S_new1 - S_old)**2) - np.sum((S_new2 - S_old)**2)
                            S_adj_prin_comp = S_set1 if S_ref_ind <= 0 else S_set2
                            if self.verbose:
                                print self.normalize(S_adj_prin_comp * (self.reputation / np.mean(self.reputation)).T)
                            net_adj_prin_comp = S_adj_prin_comp
                            convergence = not any(np.isnan(net_adj_prin_comp))
                    except:
                        continue

        elif self.algorithm == "ica-prewhitened":
            ica = FastICA(n_components=self.num_events, whiten=True)
            while not convergence:
                with warnings.catch_warnings(record=True) as w:
                    try:
                        S_ = ica.fit_transform(covariance_matrix)   # Reconstruct signals
                        if len(w):
                            continue
                        else:
                            if self.verbose:
                                print "ICA loadings:"
                                print S_
                            S_first_loading = S_[:,0]
                            S_first_loading /= np.sqrt(np.sum(S_first_loading**2))
                            S_first_score = np.array(np.dot(mean_deviation, S_first_loading)).ravel()

                            S_set1 = S_first_score + np.abs(np.min(S_first_score))
                            S_set2 = S_first_score - np.max(S_first_score)
                            S_old = np.dot(self.reputation.T, reports_filled)
                            S_new1 = np.dot(self.normalize(S_set1), reports_filled)
                            S_new2 = np.dot(self.normalize(S_set2), reports_filled)
                            S_ref_ind = np.sum((S_new1 - S_old)**2) - np.sum((S_new2 - S_old)**2)
                            S_adj_prin_comp = S_set1 if S_ref_ind <= 0 else S_set2
                            if self.verbose:
                                print self.normalize(S_adj_prin_comp * (self.reputation / np.mean(self.reputation)).T)
                            net_adj_prin_comp = S_adj_prin_comp
                            convergence = not any(np.isnan(net_adj_prin_comp))
                    except:
                        continue

        elif self.algorithm == "ica-inverse":
            ica = FastICA(n_components=self.num_events, whiten=True)
            while not convergence:
                with warnings.catch_warnings(record=True) as w:
                    try:
                        S_ = ica.fit_transform(covariance_matrix)
                        if len(w):
                            continue
                        else:
                            S_first_loading = S_[:,0]
                            S_first_loading /= np.sqrt(np.sum(S_first_loading**2))
                            S_first_score = np.array(np.dot(mean_deviation, S_first_loading)).ravel()

                            # Normalized absolute inverse scores
                            net_adj_prin_comp = 1 / np.abs(S_first_score)
                            net_adj_prin_comp /= np.sum(net_adj_prin_comp)

                            convergence = not any(np.isnan(net_adj_prin_comp))
                    except:
                        continue

        # Sum over all events in the ballot; the ratio of this sum to
        # the total covariance (over all events, across all reporters)
        # is each reporter's contribution to the overall variability.
        elif self.algorithm == "covariance":
            row_mean = np.mean(reports_filled, axis=1)
            centered = np.zeros(reports_filled.shape)
            onesvect = np.ones(self.num_events)
            for i in range(self.num_reporters): centered[i,:] = reports_filled[i,:] - onesvect * row_mean[i]

            # Unweighted: np.dot(centered, centered.T) / self.num_events
            covmat = np.dot(centered, np.ma.multiply(centered.T, self.reputation)) / float(1 - np.sum(self.reputation**2))

            # Sum across columns of the (other) covariance matrix
            contrib = np.sum(covmat, 1)
            relative_contrib = contrib / np.sum(contrib)

            set1 = relative_contrib + np.abs(np.min(relative_contrib))
            set2 = relative_contrib - np.max(relative_contrib)
            old = np.dot(self.reputation.T, reports_filled)
            new1 = np.dot(self.normalize(set1), reports_filled)
            new2 = np.dot(self.normalize(set2), reports_filled)
            ref_ind = np.sum((new1 - old)**2) - np.sum((new2 - old)**2)
            net_adj_prin_comp = set1 if ref_ind <= 0 else set2

            convergence = True

        elif self.algorithm == "covariance-unweighted":
            row_mean = np.mean(reports_filled, axis=1)
            centered = np.zeros(reports_filled.shape)
            onesvect = np.ones(self.num_events)
            for i in range(self.num_reporters): centered[i,:] = reports_filled[i,:] - onesvect * row_mean[i]

            covmat = np.dot(centered, centered.T) / self.num_events

            # Sum across columns of the (other) covariance matrix
            contrib = np.sum(covmat, 1)
            relative_contrib = contrib / np.sum(contrib)

            set1 = relative_contrib + np.abs(np.min(relative_contrib))
            set2 = relative_contrib - np.max(relative_contrib)
            old = np.dot(self.reputation.T, reports_filled)
            new1 = np.dot(self.normalize(set1), reports_filled)
            new2 = np.dot(self.normalize(set2), reports_filled)
            ref_ind = np.sum((new1 - old)**2) - np.sum((new2 - old)**2)
            net_adj_prin_comp = set1 if ref_ind <= 0 else set2

            convergence = True

        # Brute force scoring: replicated rows
        elif self.algorithm == "covariance-replicate":
            B = []
            for i in range(self.num_reporters):
                for j in range(self.reptokens[i]):
                    B.append(reports_filled[i,:].tolist())
            num_rows = len(B)
            B = np.array(B)
            row_mean = np.mean(B, axis=1)
            centered = np.zeros(B.shape)
            onesvect = np.ones(self.num_events)
            for i in range(num_rows):
                centered[i,:] = B[i,:] - onesvect * row_mean[i]
            covmat = np.dot(centered, centered.T) / self.num_events

            # Sum across columns of the (other) covariance matrix
            contrib_rpl = np.sum(covmat, 1)
            # relative_contrib_rpl = contrib_rpl / np.sum(contrib_rpl)
            relative_contrib = np.zeros(self.num_reporters)
            row = 0
            for i in range(self.num_reporters):
                relative_contrib[i] = self.reptokens[i] * contrib_rpl[row]
                # relative_contrib[i] = contrib_rpl[row] # this gives the same result as "covariance"
                row += self.reptokens[i]
            relative_contrib /= np.sum(relative_contrib)

            # import ipdb; ipdb.set_trace()

            set1 = relative_contrib + np.abs(np.min(relative_contrib))
            set2 = relative_contrib - np.max(relative_contrib)
            old = np.dot(self.reputation.T, reports_filled)
            new1 = np.dot(self.normalize(set1), reports_filled)
            new2 = np.dot(self.normalize(set2), reports_filled)
            ref_ind = np.sum((new1 - old)**2) - np.sum((new2 - old)**2)
            net_adj_prin_comp = set1 if ref_ind <= 0 else set2

            convergence = True            

        # Sum over all events in the ballot; the ratio of this sum to
        # the total cokurtosis is that reporter's contribution.
        elif self.algorithm == "cokurtosis":
            if self.aux is not None and "cokurt" in self.aux:
                cokurt_contrib = self.aux["cokurt"]
                # contrib = np.sum(np.sum(np.sum(cokurt, axis=0), axis=0), axis=0)
                # relative_contrib = cokurt_contrib / np.sum(cokurt_contrib)
                set1 = cokurt_contrib + np.abs(np.min(cokurt_contrib))
                set2 = cokurt_contrib - np.max(cokurt_contrib)
                old = np.dot(self.reputation.T, reports_filled)
                new1 = np.dot(self.normalize(set1), reports_filled)
                new2 = np.dot(self.normalize(set2), reports_filled)
                ref_ind = np.sum((new1 - old)**2) - np.sum((new2 - old)**2)
                net_adj_prin_comp = set1 if ref_ind <= 0 else set2
                convergence = True

        row_reward_weighted = self.reputation
        if max(abs(net_adj_prin_comp)) != 0:
            row_reward_weighted = self.normalize(net_adj_prin_comp * (self.reputation / np.mean(self.reputation)).T)

        smooth_rep = self.alpha*row_reward_weighted + (1-self.alpha)*self.reputation.T

        return {
            "first_loading": first_loading,
            "old_rep": self.reputation.T,
            "this_rep": row_reward_weighted,
            "smooth_rep": smooth_rep,
            "convergence": convergence,
        }

    def consensus(self):

        # Handle missing values
        reports_filled = self.interpolate(self.reports)

        # Consensus - Row Players
        # New Consensus Reward
        player_info = self.weighted_pca(reports_filled)
        adj_first_loadings = player_info['first_loading']

        # Column Players (The Event Creators)
        # Calculation of Reward for Event Authors
        # Consensus - "Who won?" Event Outcome    
        # Simple matrix multiplication ... highest information density at reporter_bonus,
        # but need EventOutcomes.Raw to get to that
        outcomes_raw = np.dot(player_info['smooth_rep'], reports_filled).squeeze()

        # Discriminate Based on Contract Type
        if self.event_bounds is not None:
            for i in range(reports_filled.shape[1]):
                # Our Current best-guess for this Scaled Event (weighted median)
                if self.event_bounds[i]["scaled"]:
                    outcomes_raw[i] = weighted_median(reports_filled[:,i].data, weights=player_info["smooth_rep"].ravel().data)

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

        # Participation
        # Information about missing values
        na_mat = self.reports * 0
        na_mat[na_mat.mask] = 1  # indicator matrix for missing

        # Participation Within Events (Columns)
        # % of reputation that answered each Event
        participation_columns = 1 - np.dot(player_info['smooth_rep'], na_mat)

        # Participation Within Agents (Rows)
        # Many options
        # 1- Democracy Option - all Events treated equally.
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
                'adj_first_loadings': adj_first_loadings.data.tolist(),
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
            'convergence': player_info['convergence'],
            'components': self.num_components,
        }

def main(argv=None):
    if argv is None:
        argv = sys.argv
    try:
        short_opts = 'hxms'
        long_opts = ['help', 'example', 'missing', 'scaled']
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
            # # new: true=1, false=-1, indeterminate=0.5, no response=0
            reports = np.array([[  1,  1, -1, -1],
                                [  1, -1, -1, -1],
                                [  1,  1, -1, -1],
                                [  1,  1,  1, -1],
                                [ -1, -1,  1,  1],
                                [ -1, -1,  1,  1]])
            reputation = [2, 10, 4, 2, 7, 1]
            oracle = Oracle(reports=reports, reputation=reputation, algorithm="covariance-replicate")
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
            oracle = Oracle(reports=reports, reputation=reputation)
            A = oracle.consensus()
            print(pd.DataFrame(A["events"]))
            print(pd.DataFrame(A["agents"]))
        elif opt in ('-s', '--scaled'):
            # reports = np.array([[ 1,  1, -1, -1 ],
            #                     [ 1, -1, -1, -1 ],
            #                     [ 1,  1, -1, -1 ],
            #                     [ 1,  1,  1, -1 ],
            #                     [-1, -1,  1,  1 ],
            #                     [-1, -1,  1,  1 ]])
            # reputation = [1, 1, 1, 1, 1, 1]
            # oracle = Oracle(reports=reports)
            # A = oracle.consensus()
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
