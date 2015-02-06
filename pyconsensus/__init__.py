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
__version__    = "0.3"
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
                 run_ica_prewhitened=False, run_ica_inverse_scores=False):
        """
        Args:
          reports (list-of-lists): reports matrix; rows = reporters, columns = Events.
          event_bounds (list): list of dicts for each Event
            {
              scaled (bool): True if scalar, False if binary (boolean)
              min (float): minimum allowed value (0 if binary)
              max (float): maximum allowed value (1 if binary)
            }

        """
        self.reports = np.ma.masked_array(reports, np.isnan(reports))
        self.event_bounds = event_bounds
        self.catch_tolerance = catch_tolerance
        self.max_row = max_row
        self.alpha = alpha
        self.verbose = verbose
        self.num_reports = len(reports)
        self.num_events = len(reports[0])
        self.run_ica = run_ica
        self.run_fixed_threshold = run_fixed_threshold
        self.run_inverse_scores = run_inverse_scores
        self.run_ica_inverse_scores = run_ica_inverse_scores
        self.run_ica_prewhitened = run_ica_prewhitened
        if reputation is None:
            self.weighted = False
            self.total_rep = self.num_reports
            self.reputation = np.array([1 / float(self.num_reports)] * self.num_reports)
            self.rep_coins = (np.copy(self.reputation) * 10**6).astype(int)
            self.total_rep_coins = sum(self.rep_coins)
        else:
            self.weighted = True
            self.total_rep = sum(np.array(reputation).ravel())
            self.reputation = np.array([i / float(self.total_rep) for i in reputation])
            self.rep_coins = (np.abs(np.copy(reputation)) * 10**6).astype(int)
            self.total_rep_coins = sum(self.rep_coins)

    def get_weight(self, v):
        """Takes an array, and returns proportional distance from zero."""
        v = abs(v)
        if np.sum(v) == 0:
            v += 1
        return v / np.sum(v)

    # def catch(self, X):
    #     """Forces continuous values into bins at 0, 0.5, and 1"""
    #     center = 0.5
    #     # print X, " vs ", center, "+/-", center + self.catch_tolerance
    #     if X < center - self.catch_tolerance:
    #         return 0
    #     elif X > center + self.catch_tolerance:
    #         return 1
    #     else:
    #         return 0.5

    def catch(self, X):
        """Forces continuous values into bins at -1, 0, and 1"""
        center = 0
        # print X, " vs ", center, "+/-", center + self.catch_tolerance
        if X < center - self.catch_tolerance:
            return -1
        elif X > center + self.catch_tolerance:
            return 1
        else:
            return 0

    def weighted_cov(self, reports_filled):
        """Weights are the number of coins people start with, so the aim of this
        weighting is to count 1 report for each of their coins -- e.g., guy with 10
        coins effectively gets 10 reports, guy with 1 coin gets 1 report, etc.

        """
        # Compute the weighted mean (of all reporters) for each event
        weighted_mean = np.ma.average(reports_filled,
                                      axis=0,
                                      weights=self.reputation.tolist())

        if self.verbose:
            print('=== INPUTS ===')
            print(reports_filled.data)
            print(self.reputation)

            print('=== WEIGHTED MEANS ===')
            print(weighted_mean)

        # Each report's difference from the mean of its event (column)
        mean_deviation = np.matrix(reports_filled - weighted_mean)

        if self.verbose:
            print('=== WEIGHTED CENTERED DATA ===')
            print(mean_deviation)

        # Compute the unbiased weighted population covariance
        # (for uniform weights, equal to np.cov(reports_filled.T, bias=1))
        ssq = np.sum(self.reputation**2)
        covariance_matrix = 1/float(1 - ssq) * np.ma.multiply(mean_deviation.T, self.reputation).dot(mean_deviation)

        if self.verbose:
            print('=== WEIGHTED COVARIANCES ===')
            print(covariance_matrix)

        return covariance_matrix, mean_deviation

    def get_reward_weights(self, reports_filled):
        """Calculates new reputations using a weighted Principal Component
        Analysis (PCA).
        
        The reports matrix has reporters as rows and events as columns.

        """
        covariance_matrix, mean_deviation = self.weighted_cov(reports_filled)

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

        if self.verbose:
            print "PCA loadings:"
            print H

        convergence = False

        if self.run_fixed_threshold:
            threshold = 0.95

            U, Sigma, Vt = np.linalg.svd(covariance_matrix)
            variance_explained = np.cumsum(Sigma / np.trace(covariance_matrix))
            
            for i, var_exp in enumerate(variance_explained):
                loading = U.T[i]
                score = np.dot(mean_deviation, loading)
                if i == 0:
                    net_score = Sigma[i] * score
                else:
                    net_score += Sigma[i] * score
                if var_exp > threshold: break

            set1 = net_score + np.abs(np.min(net_score))
            set2 = net_score - np.max(net_score)
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
            ica = FastICA(n_components=self.num_events, whiten=True)
            # ica = FastICA(n_components=self.num_events,
            #               whiten=True,
            #               random_state=0,
            #               max_iter=1000)
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
            # ica = FastICA(n_components=self.num_events, whiten=False)
            # ica = FastICA(n_components=self.num_events,
            #               whiten=True,
            #               random_state=0,
            #               max_iter=1000)
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

        if self.verbose:
            print('=== FROM SINGULAR VALUE DECOMPOSITION OF WEIGHTED COVARIANCE MATRIX ===')
            print(pd.DataFrame(SVD[0].data))
            pprint(SVD[1])
            print(pd.DataFrame(SVD[2].data))

            print('=== FIRST EIGENVECTOR ===')
            print(first_loading)

            print('=== FIRST SCORES ===')
            print(first_score)

        smooth_rep = self.alpha*row_reward_weighted + (1-self.alpha)*self.reputation.T
        return {
            "first_loading": first_loading,
            "old_rep": self.reputation.T,
            "this_rep": row_reward_weighted,
            "smooth_rep": smooth_rep,
            "convergence": convergence,
        }

    def interpolate(self, reports):
        """Uses existing data and reputations to fill missing observations.
        Essentially a weighted average using all availiable non-NA data.
        """
        reports_new = np.ma.copy(reports)
        if reports.mask.any():
            
            # Our best guess for the Event state (FALSE=0, Ambiguous=.5, TRUE=1)
            # so far (ie, using the present, non-missing, values).
            outcomes_raw = []
            for i in range(reports.shape[1]):
                
                # The Reputation of the rows (players) who DID provide
                # judgements, rescaled to sum to 1.
                active_rep = self.reputation[-reports[:,i].mask]
                active_rep[np.isnan(active_rep)] = 0
                total_active_rep = np.sum(active_rep)
                active_rep /= np.sum(active_rep)

                # The relevant Event with NAs removed.
                # ("What these row-players had to say about the Events
                # they DID judge.")
                active_events = reports[-reports[:,i].mask, i]

                # Guess the outcome; discriminate based on contract type.
                if not self.event_bounds[i]["scaled"]:
                    outcome_guess = np.dot(active_events, active_rep)
                else:
                    outcome_guess = weighted_median(active_events.data, weights=active_rep)
                outcomes_raw.append(outcome_guess)

            # Fill in the predictions to the original M
            na_mat = reports.mask  # Defines the slice of the matrix which needs to be edited.
            reports_new[na_mat] = 0  # Erase the NA's

            # Slightly complicated:
            NAsToFill = np.dot(na_mat, np.diag(outcomes_raw))
            # This builds a matrix whose columns j:
            #   na_mat was false (the observation wasn't missing) - have a value of Zero
            #   na_mat was true (the observation was missing)     - have a value of the jth element of EventOutcomes.Raw (the 'current best guess')

            reports_new += NAsToFill

            # This replaces the NAs, which were zeros, with the predicted Event outcome.

            # Appropriately force the predictions into their discrete
            # slot. (continuous variables can be gamed).
            rows, cols = reports_new.shape
            for i in range(rows):
                for j in range(cols):
                    if not self.event_bounds[j]["scaled"]:
                        reports_new[i][j] = self.catch(reports_new[i][j])

        return reports_new

    def consensus(self):
        """PCA-based consensus algorithm.

        Returns:
          dict: consensus results

        """
        scaled_reports = np.ma.copy(self.reports)
        if self.event_bounds is not None:
            # Forces a matrix of raw (user-supplied) information
            # (for example, # of House Seats, or DJIA) to conform to
            # SVD-appropriate range.
            #
            # Practically, this is done by subtracting min and dividing by
            # scaled-range (which itself is max-min).
            for i in range(self.num_events):
                if self.event_bounds[i]["scaled"]:
                    scaled_reports[:,i] = (scaled_reports[:,i] - self.event_bounds[i]["min"]) / float(self.event_bounds[i]["max"] - self.event_bounds[i]["min"])

        # Handle missing values
        reports_filled = self.interpolate(scaled_reports)
        # print("reports filled:")
        # print(reports_filled.data)

        # Consensus - Row Players
        # New Consensus Reward
        player_info = self.get_reward_weights(reports_filled)
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
                    outcomes_raw[i] = weighted_median(reports_filled[:,i],
                                                      player_info["smooth_rep"].ravel())

        # The Outcome (Discriminate Based on Contract Type)
        outcome_adj = []
        for i, raw in enumerate(outcomes_raw):
            outcome_adj.append(self.catch(raw))
            if self.event_bounds is not None and self.event_bounds[i]["scaled"]:
                outcome_adj[i] = raw

        outcome_final = []
        for i, raw in enumerate(outcomes_raw):
            outcome_final.append(outcome_adj[i])
            if self.event_bounds is not None and self.event_bounds[i]["scaled"]:
                outcome_final[i] *= self.event_bounds[i]["max"] - self.event_bounds[i]["min"]
                outcome_final[i] += self.event_bounds[i]["min"]

        # .5 is obviously undesireable, this function travels from 0 to 1
        # with a minimum at .5
        certainty = []
        for i, adj in enumerate(outcome_adj):
            certainty.append(sum(player_info["smooth_rep"][reports_filled[:,i] == adj]))
        certainty = np.array(certainty)

        # Grading Authors on a curve.
        consensus_reward = self.get_weight(certainty)

        # How well did beliefs converge?
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
                'outcome_adjusted': outcome_adj,
                'outcome_final': outcome_final,
                },
            'participation': 1 - percent_na,
            'avg_certainty': avg_certainty,
            'convergence': player_info['convergence'],
        }

def main(argv=None):
    np.set_printoptions(linewidth=500)
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
            # reports = np.array([[ 1,  1,  0,  0, 233, 16027.59],
            #                     [ 1,  0,  0,  0, 199,   np.nan],
            #                     [ 1,  1,  0,  0, 233, 16027.59],
            #                     [ 1,  1,  1,  0, 250,   np.nan],
            #                     [ 0,  0,  1,  1, 435,  8001.00],
            #                     [ 0,  0,  1,  1, 435, 19999.00]])
            # event_bounds = [
            #     {"scaled": False, "min": 0, "max": 1},
            #     {"scaled": False, "min": 0, "max": 1},
            #     {"scaled": False, "min": 0, "max": 1},
            #     {"scaled": False, "min": 0, "max": 1},
            #     {"scaled": True, "min": 0, "max": 435},
            #     {"scaled": True, "min": 8000, "max": 20000},
            # ]
            oracle = Oracle(reports=reports, event_bounds=event_bounds)
            A = oracle.consensus()
            print(pd.DataFrame(A["events"]))
            print(pd.DataFrame(A["agents"]))

if __name__ == '__main__':
    sys.exit(main(sys.argv))
