#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Consensus mechanism for Augur/Truthcoin.

pyconsensus is a Python implementation of the Augur/Truthcoin consensus
mechanism, described in detail at https://github.com/psztorc/Truthcoin.

Usage:

    from pyconsensus import Oracle

    # Example vote matrix:
    #   - each row represents a voter
    #   - each column represents a event in a prediction market
    my_votes = [[1, 1, 0, 0],
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [0, 0, 1, 1],
                [0, 0, 1, 1]]
    my_event_bounds = [
        {"scaled": True, "min": 0.1, "max": 0.5},
        {"scaled": True, "min": 0.2, "max": 0.7},
        {"scaled": False, "min": 0, "max": 1},
        {"scaled": False, "min": 0, "max": 1},
    ]

    oracle = Oracle(votes=my_votes, event_bounds=my_event_bounds)
    oracle.consensus()

"""
from __future__ import division, absolute_import
import sys
import os
import getopt
import numpy as np
from weightedstats import weighted_median
from six.moves import xrange as range

__title__      = "pyconsensus"
__version__    = "0.2.3"
__author__     = "Paul Sztorc and Jack Peterson"
__license__    = "GPL"
__maintainer__ = "Jack Peterson"
__email__      = "jack@tinybike.net"

class Oracle(object):

    def __init__(self, votes=None, event_bounds=None, reputation=None,
                 catch_tolerance=0.1, max_row=5000, alpha=0.1, verbose=False):
        """
        Args:
          votes (list-of-lists): votes matrix; rows = voters, columns = Events.
          event_bounds (list): list of dicts for each Event
            {
              scaled (bool): True if scalar, False if binary (boolean)
              min (float): minimum allowed value (0 if binary)
              max (float): maximum allowed value (1 if binary)
            }

        """
        self.votes = np.ma.masked_array(votes, np.isnan(votes))
        self.event_bounds = event_bounds
        self.catch_tolerance = catch_tolerance
        self.max_row = max_row
        self.alpha = alpha
        self.verbose = verbose
        self.num_votes = len(votes)
        if reputation is None:
            self.weighted = False
            self.total_rep = self.num_votes
            self.reputation = np.array([[1 / float(self.num_votes)]] * self.num_votes)
            self.rep_coins = (np.copy(self.reputation) * 10**6).astype(int)
        else:
            self.weighted = True
            self.total_rep = sum(np.array(reputation).ravel())
            self.reputation = np.array([i / float(self.total_rep) for i in reputation])
            self.rep_coins = (np.abs(np.copy(reputation)) * 10**6).astype(int)

    def rescale(self):
        """Forces a matrix of raw (user-supplied) information
        (for example, # of House Seats, or DJIA) to conform to
        SVD-appropriate range.

        Practically, this is done by subtracting min and dividing by
        scaled-range (which itself is max-min).

        """
        # Calulate multiplicative factors
        inv_span = []
        for scale in self.event_bounds:
            inv_span.append(1 / float(scale["max"] - scale["min"]))

        # Recenter
        out_matrix = np.ma.copy(self.votes)
        cols = self.votes.shape[1]
        for i in range(cols):
            out_matrix[:,i] -= self.event_bounds[i]["min"]

        # Rescale
        out_matrix[np.isnan(out_matrix)] = np.mean(out_matrix)

        return np.dot(out_matrix, np.diag(inv_span))

    def get_weight(self, v):
        """Takes an array, and returns proportional distance from zero."""
        v = abs(v)
        if np.sum(v) == 0:
            v += 1
        return v / np.sum(v)

    def catch(self, X):
        """Forces continuous values into bins at 0, .5, and 1"""
        if X < 0.5 * (1 - self.catch_tolerance):
            return 0
        elif X > 0.5 * (1 + self.catch_tolerance):
            return 1
        else:
            return .5

    def weighted_cov(self, votes_filled):
        """Weights are the number of coins people start with, so the aim of this
        weighting is to count 1 vote for each of their coins -- e.g., guy with 10
        coins effectively gets 10 votes, guy with 1 coin gets 1 vote, etc.

        """
        # Compute the weighted mean (of all voters) for each event
        weighted_mean = np.ma.average(votes_filled,
                                      axis=0,
                                      weights=self.rep_coins.squeeze())

        # Each vote's difference from the mean of its event (column)
        mean_deviation = np.matrix(votes_filled - weighted_mean)

        # Compute the unbiased weighted population covariance
        # (for uniform weights, equal to np.cov(votes_filled.T, bias=1))
        covariance_matrix = 1/float(np.sum(self.rep_coins)-1) * np.ma.multiply(mean_deviation, self.rep_coins).T.dot(mean_deviation)

        return covariance_matrix, mean_deviation

    def weighted_prin_comp(self, votes_filled):
        """Principal Component Analysis (PCA) on the votes matrix.

        The votes matrix has voters as rows and events as columns.

        """
        covariance_matrix, mean_deviation = self.weighted_cov(votes_filled)
        U = np.linalg.svd(covariance_matrix)[0]
        first_loading = U.T[0]
        first_score = np.dot(mean_deviation, U).T[0]
        return first_loading, first_score

    def get_reward_weights(self, votes_filled):
        """Calculates new reputations using a weighted
        Principal Component Analysis (PCA).

        """
        results = self.weighted_prin_comp(votes_filled)
        
        # The first loading (largest eigenvector) is designed to indicate
        # which Events were more 'agreed-upon' than others.
        first_loading = results[0]
        
        # The scores show loadings on consensus (to what extent does
        # this observation represent consensus?)
        first_score = results[1]

        # Which of the two possible 'new' reputation vectors had more opinion in common
        # with the original 'old' reputation.
        set1 = first_score + abs(min(first_score))
        set2 = first_score - max(first_score)
        old = np.dot(self.rep_coins.T, votes_filled)
        new1 = np.dot(self.get_weight(set1), votes_filled)
        new2 = np.dot(self.get_weight(set2), votes_filled)

        # Difference in sum of squared errors. If > 0, then new1 had higher
        # errors (use new2); conversely if < 0, then use new1.
        ref_ind = np.sum((new1 - old)**2) - np.sum((new2 - old)**2)
        if ref_ind <= 0:
            adj_prin_comp = set1
        if ref_ind > 0:
            adj_prin_comp = set2
      
        # (set this to uniform if you want a passive diffusion toward equality
        # when people cooperate [not sure why you would]). Instead diffuses towards
        # previous reputation (Smoothing does this anyway).
        row_reward_weighted = self.reputation
        if max(abs(adj_prin_comp)) != 0:
            # Overwrite the inital declaration IFF there wasn't perfect consensus.
            row_reward_weighted = self.get_weight(adj_prin_comp * (self.reputation / np.mean(self.reputation)).T)

        #note: reputation/mean(reputation) is a correction ensuring Reputation is additive. Therefore, nothing can be gained by splitting/combining Reputation into single/multiple accounts.
              
        # Freshly-Calculated Reward (Reputation) - Exponential Smoothing
        # New Reward: row_reward_weighted
        # Old Reward: reputation
        smooth_rep = self.alpha*row_reward_weighted + (1-self.alpha)*self.reputation.T

        return {
            "first_loading": first_loading,
            "old_rep": self.reputation.T,
            "this_rep": row_reward_weighted,
            "smooth_rep": smooth_rep,
        }

    def get_event_outcomes(self, votes, scaled_index):
        """Determines the Outcomes of Events based on the provided
        reputation (weighted vote).

        """
        event_outcomes_raw = []
        
        # Iterate over events (columns)
        for i in range(votes.shape[1]):

            # The Reputation of the rows (players) who DID provide
            # judgements, rescaled to sum to 1.
            active_players_rep = self.reputation * -votes[:,i].mask
            total_active_rep = np.sum(active_players_rep)

            # Normalize
            active_players_rep /= np.sum(active_players_rep)

            # The relevant Event with NAs removed.
            # ("What these row-players had to say about the Events
            # they DID judge.")
            # active_events = votes[-votes[:,i].mask, i]
            active_events = votes[:,i] * -votes[:,i].mask

            # Discriminate based on contract type.
            # Current best-guess for this Binary Event (weighted average)
            if not scaled_index[i]:
                event_outcomes_raw.append(np.dot(active_events, active_players_rep))

            # Current best-guess for this Scaled Event (weighted median)
            else:
                wmed = weighted_median(active_events, active_players_rep)
                event_outcomes_raw.append(wmed)

        return np.array(event_outcomes_raw).T

    def fill_na(self, votes, scaled_index):
        """Uses existing data and reputations to fill missing observations.
        Essentially a weighted average using all availiable non-NA data.
        """
        votes_new = np.ma.copy(votes)

        # Of course, only do this process if there ARE missing values.
        if votes.mask.any():

            # Our best guess for the Event state (FALSE=0, Ambiguous=.5, TRUE=1)
            # so far (ie, using the present, non-missing, values).
            event_outcomes_raw = self.get_event_outcomes(votes, scaled_index).squeeze()

            # Fill in the predictions to the original M
            na_mat = votes.mask  # Defines the slice of the matrix which needs to be edited.
            votes_new[na_mat] = 0  # Erase the NA's

            # Slightly complicated:
            NAsToFill = np.dot(na_mat, np.diag(event_outcomes_raw))
            # This builds a matrix whose columns j:
            #   na_mat was false (the observation wasn't missing) - have a value of Zero
            #   na_mat was true (the observation was missing)     - have a value of the jth element of EventOutcomes.Raw (the 'current best guess')

            votes_new += NAsToFill
            # This replaces the NAs, which were zeros, with the predicted Event outcome.

            # Appropriately force the predictions into their discrete
            # (0,.5,1) slot. (continuous variables can be gamed).
            rows, cols = votes_new.shape
            for i in range(rows):
                for j in range(cols):
                    if not scaled_index[j]:
                        votes_new[i][j] = self.catch(votes_new[i][j])

        return votes_new

    def consensus(self):
        """PCA-based consensus algorithm.

        Returns:
          dict: consensus results

        """
        # Fill the default scales (binary) if none are provided.
        if self.event_bounds is None:
            scaled_index = [False] * self.votes.shape[1]
            scaled_votes = self.votes
        else:
            scaled_index = [scale["scaled"] for scale in self.event_bounds]
            scaled_votes = self.rescale()

        # Handle missing values
        votes_filled = self.fill_na(scaled_votes, scaled_index)

        # Consensus - Row Players
        # New Consensus Reward
        player_info = self.get_reward_weights(votes_filled)
        adj_first_loadings = player_info['first_loading']

        # Column Players (The Event Creators)
        # Calculation of Reward for Event Authors
        # Consensus - "Who won?" Event Outcome    
        # Simple matrix multiplication ... highest information density at row_bonus,
        # but need EventOutcomes.Raw to get to that
        event_outcomes_raw = np.dot(player_info['smooth_rep'], votes_filled).squeeze()

        # Discriminate Based on Contract Type
        for i in range(votes_filled.shape[1]):
            # Our Current best-guess for this Scaled Event (weighted median)
            if scaled_index[i]:
                event_outcomes_raw[i] = weighted_median(votes_filled[:,i],
                                                           player_info["smooth_rep"].ravel())

        # .5 is obviously undesireable, this function travels from 0 to 1
        # with a minimum at .5
        certainty = abs(2 * (event_outcomes_raw - 0.5))

        # Grading Authors on a curve.
        consensus_reward = self.get_weight(certainty)

        # How well did beliefs converge?
        avg_certainty = np.mean(certainty)

        # The Outcome (Discriminate Based on Contract Type)
        event_outcome_adj = []
        for i, raw in enumerate(event_outcomes_raw):
            event_outcome_adj.append(self.catch(raw))
            if scaled_index[i]:
                event_outcome_adj[i] = raw

        event_outcome_final = []
        for i, raw in enumerate(event_outcomes_raw):
            event_outcome_final.append(event_outcome_adj[i])
            if scaled_index[i]:
                event_outcome_final[i] *= (self.event_bounds[i]["max"] - self.event_bounds[i]["min"])

        # Participation
        # Information about missing values
        na_mat = self.votes * 0
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
        na_bonus_rows = self.get_weight(participation_rows)
        row_bonus = na_bonus_rows * percent_na + player_info['smooth_rep'] * (1 - percent_na)

        # Column
        na_bonus_columns = self.get_weight(participation_columns)
        col_bonus = na_bonus_columns * percent_na + consensus_reward * (1 - percent_na)

        return {
            'original': self.votes.base,
            'filled': votes_filled.base,
            'agents': {
                'old_rep': player_info['old_rep'][0],
                'this_rep': player_info['this_rep'][0],
                'smooth_rep': player_info['smooth_rep'][0],
                'na_row': na_mat.sum(axis=1).base,
                'participation_rows': participation_rows.base,
                'relative_part': na_bonus_rows.base,
                'row_bonus': row_bonus.base,
                },
            'events': {
                'adj_first_loadings': adj_first_loadings,
                'event_outcomes_raw': event_outcomes_raw,
                'consensus_reward': consensus_reward,
                'certainty': certainty,
                'NAs Filled': na_mat.sum(axis=0),
                'participation_columns': participation_columns,
                'Author Bonus': col_bonus,
                'event_outcome_final': event_outcome_final,
                },
            'participation': 1 - percent_na,
            'certainty': avg_certainty,
        }


def main(argv=None):
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
            votes = [[1, 1, 0, 0],
                     [1, 0, 0, 0],
                     [1, 1, 0, 0],
                     [1, 1, 1, 0],
                     [0, 0, 1, 1],
                     [0, 0, 1, 1]]
            oracle = Oracle(votes=votes)
            oracle.consensus()
        elif opt in ('-m', '--missing'):
            votes = [[1, 1, 0, np.nan],
                     [1, 0, 0, 0],
                     [1, 1, 0, 0],
                     [1, 1, 1, 0],
                     [np.nan, 0, 1, 1],
                     [0, 0, 1, 1]]
            reputation = [2, 10, 4, 2, 7, 1]
            oracle = Oracle(votes=votes, reputation=reputation)
            oracle.consensus()

if __name__ == '__main__':
    sys.exit(main(sys.argv))
