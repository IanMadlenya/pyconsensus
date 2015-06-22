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
import warnings; warnings.simplefilter('ignore')
from collections import Counter
from pprint import pprint
from copy import deepcopy
import numpy as np
import pandas as pd
from scipy import cluster
from weightedstats import weighted_median
from six.moves import xrange as range
from numpy import *
import logging

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

def fold(arr, num_cols):
    folded = []
    num_rows = len(arr) / float(num_cols)
    if num_rows != int(num_rows):
        raise Exception("array length (%i) not divisible by %i" % (len(arr), num_cols))
    num_rows = int(num_rows)
    for i in range(num_rows):
        row = []
        for j in range(num_cols):
            row.append(arr[i*num_cols + j])
        folded.append(row)
    return folded

class clusternode:
    def __init__(self,vec,numItems=0,meanVec=None,rep=0,repVec=None, reporterIndexVec=None, dist=-1):
        # num of events would be == len(vec[i])
        self.vec=vec
        #numitems is num reporters in this cluster
        self.numItems=numItems
        self.meanVec=meanVec
        self.rep=rep
        self.repVec=repVec
        self.reporterIndexVec = reporterIndexVec
        self.dist = dist

class Oracle(object):

    def __init__(self, reports=None, event_bounds=None, reputation=None,
                 catch_tolerance=0.1, alpha=0.1, verbose=False,
                 aux=None, algorithm="PCA", variance_threshold=0.9,
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
            self.reputation = np.array([1 / float(self.num_reports)] * self.num_reports)
        else:
            self.weighted = True
            self.total_rep = sum(np.array(reputation).ravel())
            self.reputation = np.array([i / float(self.total_rep) for i in reputation])
        self.reptokens = [int(r * 1e6) for r in self.reputation]

    def L2dist(self,v1,v2):
        return sqrt(sum((v1-v2)**2))

    def newMean(self, cmax):
        weighted = zeros([cmax.numItems, len(cmax.vec[0])]).astype(float)
        for i in range(cmax.numItems):
            weighted[i,:] = cmax.vec[i]*cmax.repVec[i]
        x = sum(weighted, axis=0)
        mean = [y / cmax.rep for y in x]
        # logging.warning(mean)
        return(mean)

    def process(self, clusters, numReporters):
        mode = None
        numInMode = 0
        for i in range(len(clusters)):
            if(clusters[i].rep > numInMode):
                numInMode = clusters[i].rep
                mode = clusters[i]
        for x in range(len(clusters)):
            clusters[x].dist = self.L2dist(mode.meanVec,clusters[x].meanVec)
        distMatrix = zeros([numReporters, 1]).astype(float)
        for x in range(len(clusters)):
            for i in range(clusters[x].numItems):
                distMatrix[clusters[x].reporterIndexVec[i]] = clusters[x].dist
        repVector = zeros([numReporters, 1]).astype(float)
        for x in range(len(distMatrix)):
            repVector[x] = 1 / ((1 + distMatrix[x])**2)
            # repVector[x] = 1 / (distMatrix[x]+.1)
        if np.isnan(repVector).any():
            import ipdb; ipdb.set_trace()
        n = self.normalize(repVector)
        return(n.flatten())

    # expects a numpy array for reports and rep vector
    def cluster(self, features, rep, distance=L2dist):
        #cluster the rows of the "features" matrix
        distances={}
        currentclustid=-1
        clusters = []
        for i in range(len(features)):
            # cmax is most similar cluster
            cmax = None
            shortestDist = 2**255
            for n in range(len(clusters)):
                dist = self.L2dist(features[i], clusters[n].meanVec)
                if dist<shortestDist:
                    cmax = clusters[n]
                    shortestDist = dist
            if(cmax!=None and self.L2dist(features[i], cmax.meanVec) < 0.50):
                cmax.vec = concatenate((cmax.vec, array([features[i]])))
                cmax.numItems += 1
                cmax.rep += rep[i]
                cmax.repVec = append(cmax.repVec, rep[i])
                cmax.meanVec = array(self.newMean(cmax))
                cmax.reporterIndexVec += [i]
            else:
                clusters.append(clusternode(array([features[i]]), 1, features[i], rep[i], array(rep[i]), [i]))
        clusters = self.process(clusters, len(features))
        return clusters

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
        reports_mask = np.zeros([self.num_reports, self.num_events])
        missing_values = 0
        reports = np.array(reports)
        num_present = np.zeros(self.num_events).astype(int)
        for i in range(self.num_events):
            for j in range(self.num_reports):
                if reports[j,i] == NA or np.isnan(reports[j,i]):
                    reports_mask[j,i] = 1
                    missing_values += 1
                else:
                    num_present[i] += 1
        reports_copy = np.copy(reports)
        if missing_values > 0:
            for i in range(self.num_events):
                if num_present[i] < self.num_reports:
                    total_active_reputation = 0
                    active_reputation = np.zeros(num_present[i])
                    active_reports = np.zeros(num_present[i])
                    active_index = 0
                    nan_indices = np.zeros(self.num_reports) + self.num_reports
                    for j in range(self.num_reports):
                        if reports_copy[j,i] != NA and not np.isnan(reports_copy[j,i]):
                            total_active_reputation += self.reputation[j]
                            active_reputation[active_index] = self.reputation[j]
                            active_reports[active_index] = reports_copy[j,i]
                            active_index += 1
                        else:
                            nan_indices[j] = j
                    if self.event_bounds is not None and self.event_bounds[i] is not None and self.event_bounds[i]["scaled"]:
                        for j in range(num_present[i]):
                            active_reputation[j] /= total_active_reputation
                        guess = weighted_median(active_reports, weights=active_reputation)
                    else:
                        guess = 0
                        for j in range(num_present[i]):
                            active_reputation[j] /= total_active_reputation
                            guess += active_reputation[j] * active_reports[j]
                        guess = self.catch(guess)
                    for j in range(self.num_reports):
                        if nan_indices[j] < self.num_reports:
                            reports_copy[nan_indices[j],i] = guess
        return reports_copy

    def wpca(self, reports_filled):
        # Compute the weighted mean (of all reporters) for each event
        weighted_mean = np.ma.average(reports_filled,
                                      axis=0,
                                      weights=self.reputation.tolist())

        # Each report's difference from the mean of its event (column)
        wcd = np.matrix(reports_filled - weighted_mean)
        # tokens = [int(r * 1e6) for r in self.reputation]

        # Compute the unbiased weighted population covariance
        covariance_matrix = np.ma.multiply(wcd.T, self.reptokens).dot(wcd) / float(np.sum(self.reptokens) - 1)

        # H is the un-normalized eigenvector matrix
        try:
            H = np.linalg.svd(covariance_matrix)[0]
        except Exception as exc:
            print exc
            H = np.ones([self.num_events, self.num_reports])

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
        if self.verbose:
            print "Reports:"
            print self.reports
            print "Total rep:"
            print self.total_rep
            print "Num reporters:"
            print self.num_reports
            print "Rep tokens:"
            print self.reptokens

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
            self.convergence = True

        elif self.algorithm == "hierarchical":
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
            # logging.warning(nc)
            self.convergence = True

        elif self.algorithm == "clusterfeck":
            weighted_mean, wcd, covariance_matrix, first_loading, first_score = self.wpca(reports_filled)
            try:
                nc = self.cluster(reports_filled, self.reptokens)
                self.convergence = True
            except Exception as exc:
                print "clusterfeck did not complete:", exc
                print "reports:"
                print self.reports
                print "reports filled:"
                print reports_filled
                print "reputation fraction:"
                print self.reputation
                print "rep tokens:"
                print self.reptokens

        elif self.algorithm == "absolute":
            weighted_mean, wcd, covariance_matrix, first_loading, first_score = self.wpca(reports_filled)
            U, Sigma, Vt = np.linalg.svd(covariance_matrix)
            net_score = np.zeros(self.num_reports)
            for i in range(self.max_components):
                loading = U.T[i]
                if loading[0] < 0:
                    loading *= -1
                score = Sigma[i] * wcd.dot(loading)
                net_score += np.abs(score)
                if self.verbose:
                    print "  Eigenvector %d:" % i, np.round(loading, 6)
                    print "  Eigenvalue %d: " % i, Sigma[i]
                    print "  Projection:    ", np.round(score, 6)
                    print "  Nonconformity:", np.round(net_score, 6)
                    print
            nc = self.nonconformity(net_score, reports_filled)

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
        na_mat[np.isnan(self.reports)] = 1  # indicator matrix for missing
        na_mat[self.reports == NA] = 1
        if self.verbose:
            print "NA Mat:"
            print na_mat
            print

        # Participation Within Events (Columns)
        # % of reputation that answered each Event
        participation_columns = 1 - np.dot(player_info['smooth_rep'], na_mat)

        # Participation Within Agents (Rows)
        # Democracy Option - all Events treated equally.
        if self.verbose:
            print "Sum:"
            print na_mat.sum(axis=1)
            print
        participation_rows = 1 - na_mat.sum(axis=1) / na_mat.shape[1]

        # General Participation
        percent_na = 1 - np.mean(participation_columns)
        if self.verbose:
            print percent_na

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
                'outcomes_raw': outcomes_raw.tolist(),
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
            testalgo = "clusterfeck"
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
            elif arg == "7":
                reports = np.array([[ YES, YES, YES, YES, YES, YES ],
                                    [ YES, YES, YES,  NO,  NO,  NO ],
                                    [  NA,  NA,  NA,  NA,  NA,  NA ]])
                                    # [ np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])
            elif arg == "8":
                reports = np.array([[ YES, YES, YES, YES, YES, YES ],
                                    [ YES, YES, YES,  NO,  NA,  NA ],
                                    [ YES, YES, YES,  NA,  NA,  NO ]])
            elif arg == "9":
                reports = np.array([[ YES, YES, YES, YES, YES, YES ],
                                    [ YES, YES, YES,  NO,  NA,  NA ],
                                    [ YES, YES, YES,  NO,  NA,  NA ]])
            elif arg == "10":
                reports = np.array([[ YES, YES, YES,  NO, YES, YES ],
                                    [ YES, YES, YES,  NO,  NA,  NA ],
                                    [ YES, YES, YES,  NO,  NA,  NA ]])
            elif arg == "11":
                reports = np.array([[ YES, YES, YES, YES, YES, YES ],
                                    [  NA,  NA,  NA,  NA,  NA,  NA ],
                                    [ YES, YES, YES,  NO,  NO,  NO ]])
            elif arg == "12":
                reports = np.array([[ YES, YES, YES,  NO,  NO,  NO ],
                                    [ YES, YES, YES,  NO,  NO,  NO ],
                                    [ YES, YES, YES,  NO,  NO,  NO ]])
            elif arg == "13":
                reports = np.array([[ YES, YES, YES,  NO,  NO,  NO ]])
            elif arg == "14":
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
            elif arg == "15":
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
            elif arg == "16":
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
            elif arg == "17":
                reports = np.array([[ YES, YES,  NO,  NO ],
                                    [ YES,  NO,  NO,  NO ],
                                    [ YES, YES,  NO,  NO ],
                                    [ YES, YES, YES,  NO ],
                                    [  NO,  NO, YES, YES ],
                                    [  NO,  NO, YES, YES ]])
            elif arg == "18":
                reports = np.array([[ YES, YES,  NO,  NO ],
                                    [ YES,  NO,  NO,  NO ],
                                    [  NA,  NA,  NA,  NA ],
                                    [  NA,  NA,  NA,  NA ],
                                    [  NA,  NA,  NA,  NA ],
                                    [  NA,  NA,  NA,  NA ],
                                    [  NA,  NA,  NA,  NA ],
                                    [  NA,  NA,  NA,  NA ],
                                    [  NA,  NA,  NA,  NA ],
                                    [  NA,  NA,  NA,  NA ],
                                    [  NA,  NA,  NA,  NA ],
                                    [  NA,  NA,  NA,  NA ],
                                    [  NA,  NA,  NA,  NA ],
                                    [  NA,  NA,  NA,  NA ],
                                    [  NA,  NA,  NA,  NA ],
                                    [  NA,  NA,  NA,  NA ]])
            oracle = Oracle(reports=reports, algorithm=testalgo)
            A = oracle.consensus()
            print(reports)
            print(pd.DataFrame(A["events"]))
            print
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
            reports = np.array([[    YES, YES,  NO,     NA],
                                [    YES,  NO,  NO,     NO],
                                [    YES, YES,  NO,     NO],
                                [    YES, YES, YES,     NO],
                                [     NA,  NO, YES,    YES],
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
                                [ YES,  NO,  NO,  NO, 199,      NA ],
                                [ YES, YES,  NO,  NO, 233, 16027.59],
                                [ YES, YES, YES,  NO, 250,      NA ],
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
