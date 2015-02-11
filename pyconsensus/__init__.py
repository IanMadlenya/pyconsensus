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
from scipy import signal
from sklearn.decomposition import FastICA, PCA
from weightedstats import weighted_median
from six.moves import xrange as range

__title__      = "pyconsensus"
__version__    = "0.4"
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
                 catch_tolerance=0.1, max_row=5000, alpha=0.1, verbose=False,
                 run_ica=False, run_fixed_threshold=False, run_inverse_scores=False,
                 run_ica_prewhitened=False, run_ica_inverse_scores=False,
                 run_fixed_threshold_sum=False, variance_threshold=0.85):
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
        self.event_bounds = event_bounds
        self.catch_tolerance = catch_tolerance
        self.max_row = max_row
        self.alpha = alpha
        self.verbose = verbose
        self.num_players = len(reports)
        self.num_events = len(reports[0])
        self.run_ica = run_ica
        self.run_fixed_threshold = run_fixed_threshold
        self.run_fixed_threshold_sum = run_fixed_threshold_sum
        if run_fixed_threshold:
            self.variance_threshold = variance_threshold
        self.run_inverse_scores = run_inverse_scores
        self.run_ica_inverse_scores = run_ica_inverse_scores
        self.run_ica_prewhitened = run_ica_prewhitened
        if reputation is None:
            self.weighted = False
            self.total_rep = self.num_players
            self.reputation = np.array([1 / float(self.num_players)] * self.num_players)
        else:
            self.weighted = True
            self.total_rep = sum(np.array(reputation).ravel())
            self.reputation = np.array([i / float(self.total_rep) for i in reputation])

    def get_weight(self, v):
        """Takes an array, and returns proportional distance from zero."""
        v = abs(v)
        if np.sum(v) == 0:
            v += 1
        return v / np.sum(v)

    def catch(self, X):
        """Forces continuous values into bins at -1, 0, and 1"""
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
                for i in range(self.num_players):
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

    def weighted_cov(self, reports_filled):
        """Weights are the number of coins people start with, so the aim of this
        weighting is to count 1 report for each of their coins -- e.g., guy with 10
        coins effectively gets 10 reports, guy with 1 coin gets 1 report, etc.

        """
        # Compute the weighted mean (of all reporters) for each event
        weighted_mean = np.ma.average(reports_filled,
                                      axis=0,
                                      weights=self.reputation.tolist())

        # Each report's difference from the mean of its event (column)
        mean_deviation = np.matrix(reports_filled - weighted_mean)

        # Compute the unbiased weighted population covariance
        # (for uniform weights, equal to np.cov(reports_filled.T, bias=1))
        ssq = np.sum(self.reputation**2)
        covariance_matrix = 1/float(1 - ssq) * np.ma.multiply(mean_deviation.T, self.reputation).dot(mean_deviation)

        return covariance_matrix, mean_deviation

    def weighted_pca(self, reports_filled):
        """Calculates new reputations using a weighted Principal Component
        Analysis (PCA).
        
        The reports matrix has reporters as rows and events as columns.

        """
        covariance_matrix, mean_deviation = self.weighted_cov(reports_filled)

        #############################
        # Reference implementation: #
        # First principal component #
        #############################

        # H is the un-normalized eigenvector matrix
        H = np.linalg.svd(covariance_matrix)[0]

        # Normalize loading by Euclidean distance
        first_loading = np.ma.masked_array(H[:,0] / np.sqrt(np.sum(H[:,0]**2)))
        first_score = np.dot(mean_deviation, first_loading)

        set1 = first_score + np.abs(np.min(first_score))
        set2 = first_score - np.max(first_score)
        old = np.dot(self.reputation.T, reports_filled)
        new1 = np.dot(self.get_weight(set1), reports_filled)
        new2 = np.dot(self.get_weight(set2), reports_filled)
        ref_ind = np.sum((new1 - old)**2) - np.sum((new2 - old)**2)
        adj_prin_comp = set1 if ref_ind <= 0 else set2

        convergence = False

        if self.run_fixed_threshold:

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

            length = np.sqrt(length)

            net_adj_prin_comp = 1 / np.abs(length)
            net_adj_prin_comp /= np.sum(net_adj_prin_comp)

            convergence = True

        elif self.run_fixed_threshold_sum:

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

            set1 = length + np.abs(np.min(length))
            set2 = length - np.max(length)
            old = np.dot(self.reputation.T, reports_filled)
            new1 = np.dot(self.get_weight(set1), reports_filled)
            new2 = np.dot(self.get_weight(set2), reports_filled)

            ref_ind = np.sum((new1 - old)**2) - np.sum((new2 - old)**2)
            net_adj_prin_comp = set1 if ref_ind <= 0 else set2

            convergence = True            

        elif self.run_inverse_scores:

            # principal_components = PCA().fit_transform(covariance_matrix)
            principal_components = np.linalg.svd(covariance_matrix)[0]

            first_loading = principal_components[:,0]
            first_loading = np.ma.masked_array(first_loading / np.sqrt(np.sum(first_loading**2)))
            first_score = np.dot(mean_deviation, first_loading)

            # Normalized absolute inverse scores
            net_adj_prin_comp = 1 / np.abs(first_score)
            net_adj_prin_comp /= np.sum(net_adj_prin_comp)

            convergence = True

        elif self.run_ica:
            ica = FastICA(n_components=self.num_events,
                          whiten=False,
                          random_state=0,
                          max_iter=1000)
            with warnings.catch_warnings(record=True) as w:
                try:
                    S_ = ica.fit_transform(covariance_matrix)   # Reconstruct signals
                    if len(w):
                        net_adj_prin_comp = adj_prin_comp
                    else:
                        if self.verbose:
                            print "ICA loadings:"
                            print S_
                        
                        S_first_loading = S_[:,0]
                        S_first_loading /= np.sqrt(np.sum(S_first_loading**2))
                        S_first_score = np.array(np.dot(mean_deviation, S_first_loading)).flatten()

                        S_set1 = S_first_score + np.abs(np.min(S_first_score))
                        S_set2 = S_first_score - np.max(S_first_score)
                        S_old = np.dot(self.reputation.T, reports_filled)
                        S_new1 = np.dot(self.get_weight(S_set1), reports_filled)
                        S_new2 = np.dot(self.get_weight(S_set2), reports_filled)

                        S_ref_ind = np.sum((S_new1 - S_old)**2) - np.sum((S_new2 - S_old)**2)
                        S_adj_prin_comp = S_set1 if S_ref_ind <= 0 else S_set2

                        if self.verbose:
                            print self.get_weight(S_adj_prin_comp * (self.reputation / np.mean(self.reputation)).T)

                        net_adj_prin_comp = S_adj_prin_comp

                        # Normalized absolute inverse scores
                        net_adj_prin_comp = 1 / np.abs(first_score)
                        net_adj_prin_comp /= np.sum(net_adj_prin_comp)

                        convergence = not any(np.isnan(net_adj_prin_comp))
                except:
                    net_adj_prin_comp = adj_prin_comp

        elif self.run_ica_prewhitened:
            ica = FastICA(n_components=self.num_events, whiten=False)
            with warnings.catch_warnings(record=True) as w:
                try:
                    S_ = ica.fit_transform(covariance_matrix)   # Reconstruct signals
                    if len(w):
                        net_adj_prin_comp = adj_prin_comp
                    else:
                        if self.verbose:
                            print "ICA loadings:"
                            print S_
                        
                        S_first_loading = S_[:,0]
                        S_first_loading /= np.sqrt(np.sum(S_first_loading**2))
                        S_first_score = np.array(np.dot(mean_deviation, S_first_loading)).flatten()

                        S_set1 = S_first_score + np.abs(np.min(S_first_score))
                        S_set2 = S_first_score - np.max(S_first_score)
                        S_old = np.dot(self.reputation.T, reports_filled)
                        S_new1 = np.dot(self.get_weight(S_set1), reports_filled)
                        S_new2 = np.dot(self.get_weight(S_set2), reports_filled)

                        S_ref_ind = np.sum((S_new1 - S_old)**2) - np.sum((S_new2 - S_old)**2)
                        S_adj_prin_comp = S_set1 if S_ref_ind <= 0 else S_set2

                        if self.verbose:
                            print self.get_weight(S_adj_prin_comp * (self.reputation / np.mean(self.reputation)).T)

                        net_adj_prin_comp = S_adj_prin_comp

                        # Normalized absolute inverse scores
                        net_adj_prin_comp = 1 / np.abs(first_score)
                        net_adj_prin_comp /= np.sum(net_adj_prin_comp)

                        convergence = not any(np.isnan(net_adj_prin_comp))
                except:
                    net_adj_prin_comp = adj_prin_comp

        elif self.run_ica_inverse_scores:
            ica = FastICA(n_components=self.num_events, whiten=True)
            with warnings.catch_warnings(record=True) as w:
                try:
                    S_ = ica.fit_transform(covariance_matrix)   # Reconstruct signals
                    if len(w):
                        net_adj_prin_comp = adj_prin_comp
                    else:
                        if self.verbose:
                            print "ICA loadings:"
                            print S_
                        
                        S_first_loading = S_[:,0]
                        S_first_loading /= np.sqrt(np.sum(S_first_loading**2))
                        S_first_score = np.array(np.dot(mean_deviation, S_first_loading)).flatten()

                        # Normalized absolute inverse scores
                        net_adj_prin_comp = 1 / np.abs(S_first_score)
                        net_adj_prin_comp /= np.sum(net_adj_prin_comp)

                        convergence = not any(np.isnan(net_adj_prin_comp))
                except:
                    net_adj_prin_comp = adj_prin_comp
        else:
            net_adj_prin_comp = adj_prin_comp

        row_reward_weighted = self.reputation
        if max(abs(net_adj_prin_comp)) != 0:
            row_reward_weighted = self.get_weight(net_adj_prin_comp * (self.reputation / np.mean(self.reputation)).T)

        smooth_rep = self.alpha*row_reward_weighted + (1-self.alpha)*self.reputation.T

        return {
            "first_loading": first_loading,
            "old_rep": self.reputation.T,
            "this_rep": row_reward_weighted,
            "smooth_rep": smooth_rep,
            "convergence": convergence,
        }

    def consensus(self):
        """PCA-based consensus algorithm.

        Returns:
          dict: consensus results

        """
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
        consensus_reward = self.get_weight(certainty)
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
        na_bonus_reporters = self.get_weight(participation_rows)
        reporter_bonus = na_bonus_reporters * percent_na + player_info['smooth_rep'] * (1 - percent_na)

        # Column
        na_bonus_events = self.get_weight(participation_columns)
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
            reports = np.array([
                [-1.0,   0.0,  -1.0,   1.0,   1.0,  -1.0,   0.0,   1.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   0.0,  -1.0,   0.0,   1.0,   1.0],   # "true"    
                [-1.0,   0.0,  -1.0,   1.0,   1.0,  -1.0,   0.0,   1.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   0.0,  -1.0,   0.0,   1.0,   1.0],   # "true"    
                [-1.0,   0.0,  -1.0,   1.0,   1.0,  -1.0,   0.0,   1.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   0.0,  -1.0,   0.0,   1.0,   1.0],   # "true"    
                [-1.0,   0.0,  -1.0,   1.0,   1.0,  -1.0,   0.0,   1.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   0.0,  -1.0,   0.0,   1.0,   1.0],   # "true"    
                [-1.0,   0.0,  -1.0,   1.0,   1.0,  -1.0,   0.0,   1.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   0.0,  -1.0,   0.0,   1.0,   1.0],   # "distort" 
                [-1.0,   0.0,  -1.0,   1.0,   1.0,  -1.0,   0.0,   1.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   0.0,  -1.0,   0.0,   1.0,   1.0],   # "true"    
                [-1.0,   0.0,  -1.0,   1.0,   1.0,  -1.0,   0.0,   1.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   0.0,  -1.0,   0.0,   1.0,   1.0],   # "distort" 
                [-1.0,   0.0,  -1.0,   1.0,   1.0,  -1.0,   0.0,   1.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   0.0,  -1.0,   0.0,   1.0,   1.0],   # "true"    
                [ 0.0,   1.0,   1.0,   1.0,  -1.0,  -1.0,  -1.0,   1.0,  -1.0,  -1.0,   1.0,   1.0,   0.0,   1.0,  -1.0,   0.0,   1.0,  -1.0,   1.0,  -1.0,  -1.0,  -1.0,  -1.0,   1.0,  -1.0],   # "liar"    
                [-1.0,   0.0,  -1.0,   1.0,   1.0,  -1.0,   0.0,   1.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   0.0,  -1.0,   0.0,   1.0,   1.0],   # "true"    
                [-1.0,   0.0,  -1.0,   1.0,   1.0,  -1.0,   0.0,   1.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   0.0,  -1.0,   0.0,   1.0,   1.0],   # "true"    
                [-1.0,   0.0,  -1.0,   1.0,   1.0,  -1.0,   0.0,   1.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   0.0,  -1.0,   0.0,   1.0,   1.0],   # "true"    
                [-1.0,   0.0,  -1.0,   1.0,   1.0,  -1.0,   0.0,   1.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   0.0,  -1.0,   0.0,   1.0,   1.0],   # "true"    
                [-1.0,   0.0,  -1.0,   1.0,   1.0,  -1.0,   0.0,   1.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   0.0,  -1.0,   0.0,   1.0,   1.0],   # "true"    
                [-1.0,   0.0,  -1.0,   1.0,   1.0,  -1.0,   0.0,   1.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   0.0,  -1.0,   0.0,   1.0,   1.0],   # "distort" 
                [-1.0,   0.0,  -1.0,   1.0,   1.0,  -1.0,   0.0,   1.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   0.0,  -1.0,   0.0,   1.0,   1.0],   # "true"    
                [-1.0,   0.0,  -1.0,   1.0,   1.0,  -1.0,   0.0,   1.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   0.0,  -1.0,   0.0,   1.0,   1.0],   # "distort" 
                [-1.0,   0.0,  -1.0,   1.0,   1.0,  -1.0,   0.0,   1.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   0.0,  -1.0,   0.0,   1.0,   1.0],   # "distort" 
                [-1.0,   0.0,  -1.0,   1.0,   1.0,  -1.0,   0.0,   1.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   0.0,  -1.0,   0.0,   1.0,   1.0],   # "true"    
                [-1.0,   0.0,  -1.0,   1.0,   1.0,  -1.0,   0.0,   1.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   0.0,  -1.0,   0.0,   1.0,   1.0],   # "distort" 
                [ 1.0,  -1.0,  -1.0,   1.0,   0.0,   1.0,   0.0,   1.0,   1.0,   0.0,   1.0,   1.0,   0.0,  -1.0,  -1.0,   1.0,   0.0,  -1.0,   1.0,  -1.0,   0.0,   1.0,   0.0,   1.0,   1.0],   # "liar"    
                [-1.0,   0.0,  -1.0,   1.0,   1.0,  -1.0,   0.0,   1.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   0.0,  -1.0,   0.0,   1.0,   1.0],   # "true"    
                [-1.0,   0.0,  -1.0,   1.0,   1.0,  -1.0,   0.0,   1.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   0.0,  -1.0,   0.0,   1.0,   1.0],   # "true"    
                [ 0.0,  -1.0,   0.0,   1.0,  -1.0,  -1.0,  -1.0,   1.0,   0.0,   0.0,   0.0,   0.0,   1.0,   1.0,   1.0,   1.0,  -1.0,   1.0,   1.0,  -1.0,   1.0,   1.0,  -1.0,   1.0,   0.0],   # "liar"    
                [ 0.0,   0.0,   0.0,  -1.0,  -1.0,   1.0,  -1.0,   1.0,   0.0,   1.0,  -1.0,  -1.0,  -1.0,   0.0,   0.0,   1.0,  -1.0,  -1.0,   0.0,  -1.0,   0.0,   0.0,   0.0,   1.0,   0.0],   # "liar"    
                [-1.0,   0.0,  -1.0,   1.0,   1.0,  -1.0,   0.0,   1.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   0.0,  -1.0,   0.0,   1.0,   1.0],   # "true"    
                [ 0.0,   0.0,  -1.0,  -1.0,   1.0,  -1.0,   0.0,   1.0,   1.0,  -1.0,   1.0,   1.0,   1.0,   0.0,   1.0,   1.0,   0.0,  -1.0,  -1.0,  -1.0,  -1.0,   0.0,   1.0,  -1.0,   0.0],   # "liar"    
                [-1.0,   0.0,  -1.0,   1.0,   1.0,  -1.0,   0.0,   1.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   0.0,  -1.0,   0.0,   1.0,   1.0],   # "true"    
                [ 1.0,  -1.0,  -1.0,  -1.0,   0.0,  -1.0,  -1.0,   1.0,   1.0,   1.0,  -1.0,   1.0,  -1.0,  -1.0,  -1.0,  -1.0,   0.0,   1.0,  -1.0,   1.0,   0.0,   0.0,   0.0,   1.0,   0.0],   # "liar"    
                [ 1.0,   1.0,   0.0,   0.0,   0.0,   1.0,   1.0,   0.0,   0.0,   1.0,  -1.0,   1.0,   1.0,  -1.0,   0.0,   1.0,   0.0,   1.0,  -1.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0],   # "liar"    
                [-1.0,   0.0,  -1.0,   1.0,   1.0,  -1.0,   0.0,   1.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   0.0,  -1.0,   0.0,   1.0,   1.0],   # "true"    
                [-1.0,   0.0,  -1.0,   1.0,   1.0,  -1.0,   0.0,   1.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   0.0,  -1.0,   0.0,   1.0,   1.0],   # "true"    
                [ 1.0,   1.0,   1.0,  -1.0,  -1.0,   1.0,  -1.0,   1.0,  -1.0,  -1.0,   0.0,   0.0,  -1.0,   0.0,   1.0,   1.0,   1.0,   0.0,   0.0,  -1.0,   1.0,   1.0,  -1.0,   0.0,   1.0],   # "liar"    
                [ 1.0,   1.0,   0.0,   0.0,   0.0,   1.0,   1.0,   0.0,   0.0,   1.0,  -1.0,   1.0,   1.0,  -1.0,   0.0,   1.0,   0.0,   1.0,  -1.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0],   # "liar"    
                [-1.0,   0.0,  -1.0,   1.0,   1.0,  -1.0,   0.0,   1.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   0.0,  -1.0,   0.0,   1.0,   1.0],   # "true"    
                [-1.0,   0.0,  -1.0,   1.0,   1.0,  -1.0,   0.0,   1.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   0.0,  -1.0,   0.0,   1.0,   1.0],   # "true"    
                [-1.0,   0.0,  -1.0,   0.0,  -1.0,  -1.0,  -1.0,  -1.0,  -1.0,   0.0,   1.0,   0.0,   1.0,   0.0,  -1.0,  -1.0,  -1.0,   1.0,   0.0,   1.0,   0.0,  -1.0,   1.0,  -1.0,   0.0],   # "liar"    
                [-1.0,   0.0,  -1.0,   1.0,   1.0,  -1.0,   0.0,   1.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   0.0,  -1.0,   0.0,   1.0,   1.0],   # "true"    
                [-1.0,   0.0,  -1.0,   0.0,  -1.0,  -1.0,  -1.0,  -1.0,  -1.0,   0.0,   1.0,   0.0,   1.0,   0.0,  -1.0,  -1.0,  -1.0,   1.0,   0.0,   1.0,   0.0,  -1.0,   1.0,  -1.0,   0.0],   # "liar"    
                [-1.0,   0.0,  -1.0,   1.0,   1.0,  -1.0,   0.0,   1.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   0.0,  -1.0,   0.0,   1.0,   1.0],   # "distort" 
                [-1.0,   0.0,  -1.0,   1.0,   1.0,  -1.0,   0.0,   1.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   0.0,  -1.0,   0.0,   1.0,   1.0],   # "true"    
                [-1.0,   0.0,  -1.0,   1.0,   1.0,  -1.0,   0.0,   1.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   0.0,  -1.0,   0.0,   1.0,   1.0],   # "distort" 
                [-1.0,   0.0,  -1.0,   1.0,   1.0,  -1.0,   0.0,   1.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   0.0,  -1.0,   0.0,   1.0,   1.0],   # "true"    
                [-1.0,   0.0,  -1.0,   1.0,   1.0,  -1.0,   0.0,   1.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   0.0,  -1.0,   0.0,   1.0,   1.0],   # "true"    
                [ 1.0,   0.0,   1.0,   1.0,  -1.0,   0.0,   1.0,   0.0,   1.0,   1.0,  -1.0,   0.0,  -1.0,   0.0,  -1.0,  -1.0,   1.0,  -1.0,  -1.0,   0.0,   0.0,   1.0,  -1.0,   0.0,  -1.0],   # "liar"    
                [-1.0,   0.0,  -1.0,   1.0,   1.0,  -1.0,   0.0,   1.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   0.0,  -1.0,   0.0,   1.0,   1.0],   # "true"    
                [-1.0,   0.0,  -1.0,   1.0,   1.0,  -1.0,   0.0,   1.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   0.0,  -1.0,   0.0,   1.0,   1.0],   # "distort" 
                [-1.0,   0.0,  -1.0,   1.0,   1.0,  -1.0,   0.0,   1.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   0.0,  -1.0,   0.0,   1.0,   1.0],   # "true"    
                [-1.0,   0.0,  -1.0,   1.0,   1.0,  -1.0,   0.0,   1.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   0.0,  -1.0,   0.0,   1.0,   1.0],   # "true"    
                [ 1.0,   0.0,   1.0,   1.0,  -1.0,   0.0,   1.0,   0.0,   1.0,   1.0,  -1.0,   0.0,  -1.0,   0.0,  -1.0,  -1.0,   1.0,  -1.0,  -1.0,   0.0,   0.0,   1.0,  -1.0,   0.0,  -1.0],   # "liar"
            ])
            # oracle = Oracle(reports=reports, reputation=reputation)
            oracle = Oracle(reports=reports, run_ica=True)
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
