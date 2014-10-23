#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Consensus mechanism for Truthcoin.

pyconsensus is a Python implementation of the Truthcoin consensus mechanism,
described in detail at https://github.com/psztorc/Truthcoin.

Usage:

    from pyconsensus import Oracle

    # Example vote matrix:
    #   - each row represents a voter
    #   - each column represents a decision in a prediction market
    my_votes = [[1, 1, 0, 0],
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [0, 0, 1, 1],
                [0, 0, 1, 1]]
    my_decision_bounds = [
        {"scaled": True, "min": 0.1, "max": 0.5},
        {"scaled": True, "min": 0.2, "max": 0.7},
        {"scaled": False, "min": 0, "max": 1},
        {"scaled": False, "min": 0, "max": 1},
    ]

    oracle = Oracle(votes=my_votes, decision_bounds=my_decision_bounds)
    oracle.consensus()

"""
from __future__ import division, print_function, unicode_literals, absolute_import
import sys
try:
    import cdecimal
    sys.modules["decimal"] = cdecimal
except:
    pass
import os
import getopt
from decimal import Decimal, getcontext, ROUND_HALF_EVEN
import numpy as np
import numpy.ma as ma
from weightedstats import weighted_median

__title__      = "pyconsensus"
__version__    = "0.2"
__author__     = "Paul Sztorc and Jack Peterson"
__license__    = "GPL"
__maintainer__ = "Jack Peterson"
__email__      = "jack@tinybike.net"

# Python 3 compatibility
from six.moves import xrange as range
_IS_PYTHON_3 = sys.version_info[0] == 3
identity = lambda x : x
if _IS_PYTHON_3:
    u = identity
else:
    import codecs
    def u(string):
        return codecs.unicode_escape_decode(string)[0]

getcontext().rounding = ROUND_HALF_EVEN

class Oracle(object):

    def __init__(self, votes=None, decision_bounds=None, weights=-1,
                 catch_p=.1, max_row=5000, verbose=False):
        self.votes = ma.masked_array(votes, np.isnan(votes))
        self.decision_bounds = decision_bounds
        self.weights = weights
        self.catch_p = catch_p
        self.max_row = max_row
        self.verbose = verbose

    def Rescale(self, UnscaledMatrix, Scales):
        """Forces a matrix of raw (user-supplied) information
        (for example, # of House Seats, or DJIA) to conform to
        svd-appropriate range.

        Practically, this is done by subtracting min and dividing by
        scaled-range (which itself is max-min).

        """
        # Calulate multiplicative factors   
        InvSpan = []
        for scale in Scales:
            InvSpan.append(1 / float(scale["max"] - scale["min"]))

        # Recenter
        OutMatrix = ma.copy(UnscaledMatrix)
        cols = UnscaledMatrix.shape[1]
        for i in range(cols):
            OutMatrix[:,i] -= Scales[i]["min"]

        # Rescale
        NaIndex = np.isnan(OutMatrix)
        OutMatrix[NaIndex] = 0

        OutMatrix = np.dot(OutMatrix, np.diag(InvSpan))
        OutMatrix[NaIndex] = np.nan

        return OutMatrix

    def MeanNa(self, Vec):
        """Takes masked array, replaces missing values with array mean."""
        MM = np.mean(Vec)
        Vec[np.where(Vec.mask)] = MM
        return Vec
        
    def GetWeight(self, Vec, AddMean=0):
        """Takes an array, and returns proportional distance from zero."""
        New = abs(Vec)

        # Add the mean to each element of the vector
        if AddMean == 1:
            New += np.mean(New)

        # Catch an error here
        if np.sum(New) == 0:
            New += 1

        # Normalize
        return New / np.sum(New)

    def Catch(self, X, Tolerance=0):
        """Forces continuous values into bins at 0, .5, and 1"""
        if X < (.5-(Tolerance/2)):
            return 0
        elif X > (.5+(Tolerance/2)):
            return 1
        else:
            return .5

    def Influence(self, Weight):
        """Takes a normalized Vector (one that sums to 1), and computes
        relative strength of the indicators.
        """
        N = len(Weight)
        Expected = [[1/N]] * N
        Out = []
        for i in range(1, N):
            Out.append(Weight[i] / Expected[i])
        return Out

    def ReWeight(self, Vec):
        """Get the relative influence of numbers, treat NaN as influence-less."""
        Out = Vec
        Exclude = np.isnan(Vec)

        # Set missing values to 0
        Out[Exclude] = 0

        # Normalize
        return Out / np.sum(Out)

    def DemocracyCoin(self, votes):
        """No coin distribution given, assuming democracy: one person, one vote."""
        return self.GetWeight(np.array([[1]] * len(votes)))

    def WeightedCov(self, votes, reputation=-1):
        """Takes 1] a masked array, and 2] an [n x 1] dimentional array of
        weights, and computes the weighted covariance matrix and center of
        a given array.
        Taken from
        http://stats.stackexchange.com/questions/61225/correct-equation-for-weighted-unbiased-sample-covariance
        """
        if type(reputation) is int:
            reputation = self.DemocracyCoin(votes)

        Coins = np.array([map(int, i) for i in reputation * 10**6])

        # Computing the weighted sample mean (fast, efficient and precise)
        weighted_sample_mean = ma.average(votes, axis=0, weights=np.hstack(Coins))
        XM = np.matrix(votes - weighted_sample_mean) # xm = X diff to mean

        # Compute the unbiased weighted sample covariance
        sigma2 = np.matrix(1/(np.sum(Coins)-1) * ma.multiply(XM, Coins).T.dot(XM));

        return {'Cov': np.array(sigma2), 'Center': np.array(XM)}

    def WeightedPrinComp(self, votes, reputation=-1):
        """Takes a matrix and row-weights and manually computes the statistical procedure known as Principal Components Analysis (PCA)
        This version of the procedure is so basic, that it can also be thought of as merely a singular-value decomposition on a weighted covariance matrix."""      
        wCVM = self.WeightedCov(votes, reputation)
        SVD = np.linalg.svd(wCVM['Cov'])

        # First loading
        L = SVD[0].T[0]

        # First score
        S = np.dot(wCVM['Center'], SVD[0]).T[0]

        return L, S

    def GetRewardWeights(self, votes, reputation=-1, Alpha=.1, Verbose=False):
        """Calculates new reputations using a weighted
        Principal Component Analysis (PCA).

        """
        if type(reputation) is int:
            reputation = self.DemocracyCoin(votes)

        Results = self.WeightedPrinComp(votes, reputation)
        # Need the old reputation back for the rest of this.
        reputation = self.GetWeight(reputation)
        # The first loading is designed to indicate which Decisions were more 'agreed-upon' than others.
        FirstLoading = Results[0]
        # The scores show loadings on consensus (to what extent does this observation represent consensus?)
        FirstScore = Results[1]

        #PCA, being an abstract factorization, is incapable of determining anything absolute.
        #Therefore the results of the entire procedure would theoretically be reversed if the average state of Decisions changed from TRUE to FALSE.
        #Because the average state of Decisions is a function both of randomness and the way the Decisions are worded, I quickly check to see which
        #  of the two possible 'new' reputation vectors had more opinion in common with the original 'old' reputation.
        #  I originally tried doing this using math but after multiple failures I chose this ad hoc way.
        Set1 = FirstScore + abs(min(FirstScore))
        Set2 = FirstScore - max(FirstScore)
        Old = np.dot(reputation.T, votes)
        New1 = np.dot(self.GetWeight(Set1), votes)
        New2 = np.dot(self.GetWeight(Set2), votes)

        # Difference in sum of squared errors. If > 0, then New1 had higher
        # errors (use New2); conversely if < 0, then use New1.
        RefInd = np.sum((New1 - Old)**2) - np.sum((New2 - Old)**2)
        if RefInd <= 0:
            AdjPrinComp = Set1
        if RefInd > 0:
            AdjPrinComp = Set2

        if Verbose:
            print("")
            print("Estimations using: Previous reputation, Option 1, Option 2")
            print(np.vstack([Old, New1, New2]).T)
            print("")
            print("Previous period reputations, Option 1, Option 2, Selection")
            print(np.vstack([reputation.T, Set1, Set2, AdjPrinComp]).T)
      
        #Declared here, filled below (unless there was a perfect consensus).
        RowRewardWeighted = reputation # (set this to uniform if you want a passive diffusion toward equality when people cooperate [not sure why you would]). Instead diffuses towards previous reputation (Smoothing does this anyway).
        if max(abs(AdjPrinComp)) != 0:
            # Overwrite the inital declaration IFF there wasn't perfect consensus.
            RowRewardWeighted = self.GetWeight(AdjPrinComp * (reputation / np.mean(reputation)).T)

        #note: reputation/mean(reputation) is a correction ensuring Reputation is additive. Therefore, nothing can be gained by splitting/combining Reputation into single/multiple accounts.
              
        # Freshly-Calculated Reward (Reputation) - Exponential Smoothing
        # New Reward: RowRewardWeighted
        # Old Reward: reputation
        SmoothedR = Alpha*RowRewardWeighted + (1-Alpha)*reputation.T

        return {
            "FirstL": FirstLoading,
            "OldRep": reputation.T,
            "ThisRep": RowRewardWeighted,
            "SmoothRep":SmoothedR,
        }

    def GetDecisionOutcomes(self, Mtemp, ScaledIndex, Rep=-1, Verbose=False):
        """Determines the Outcomes of Decisions based on the provided reputation (weighted vote)."""

        if type(Rep) is int:
            Rep = self.DemocracyCoin(Mtemp)

        RewardWeightsNA = Rep
        DecisionOutcomes_Raw = []

        # For each column:
        for i in range(Mtemp.shape[1]):

            # The Reputation of the row-players who DID provide
            # judgements, rescaled to sum to 1.
            Row = self.ReWeight(RewardWeightsNA[-Mtemp[..., i].mask])

            # The relevant Decision with NAs removed.
            # ("What these row-players had to say about the Decisions
            # they DID judge.")
            Col = Mtemp[-Mtemp[..., i].mask, i]

            # Discriminate Based on Contract Type
            if not ScaledIndex[i]:
                # Our Current best-guess for this Binary Decision (weighted average)
                DecisionOutcomes_Raw.append(np.dot(Col, Row))
            else:
                # Our Current best-guess for this Scaled Decision (weighted median)
                wmed = weighted_median(Row[:,0], Col)
                DecisionOutcomes_Raw.append(wmed)

        return np.array(DecisionOutcomes_Raw).T

    def FillNa(self, votes_na, ScaledIndex, reputation=-1, CatchP=.1, Verbose=False):
        """Uses exisiting data and reputations to fill missing observations.
        Essentially a weighted average using all availiable non-NA data.
        How much should slackers who arent voting suffer? I decided this would
        depend on the global percentage of slacking.
        """
        if type(reputation) is int:
            reputation = self.DemocracyCoin(votes_na)

        # In case no Missing values, Mnew and votes_na will be the same.
        votes_na_new = ma.copy(votes_na)

        # Of course, only do this process if there ARE missing values.
        if votes_na.mask.any():
            # Our best guess for the Decision state (FALSE=0, Ambiguous=.5, TRUE=1)
            # so far (ie, using the present, non-missing, values).
            DecisionOutcomes_Raw = self.GetDecisionOutcomes(votes_na, ScaledIndex, reputation, Verbose).squeeze()

            # Fill in the predictions to the original M
            NAmat = votes_na.mask  # Defines the slice of the matrix which needs to be edited.
            votes_na_new[NAmat] = 0  # Erase the NA's

            # Slightly complicated:
            NAsToFill = np.dot(NAmat, np.diag(DecisionOutcomes_Raw))
            
            # This builds a matrix whose columns j:
            #   NAmat was false (the observation wasn't missing) - have a value of Zero
            #   NAmat was true (the observation was missing)     - have a value of the jth element of DecisionOutcomes.Raw (the 'current best guess')
            votes_na_new += NAsToFill
            # This replaces the NAs, which were zeros, with the predicted Decision outcome.

            # Appropriately force the predictions into their discrete
            # (0,.5,1) slot. (continuous variables can be gamed).
            rows, cols = votes_na_new.shape
            for i in range(rows):
                for j in range(cols):
                    if not ScaledIndex[j]:
                        votes_na_new[i][j] = self.Catch(votes_na_new[i][j], CatchP)

        return votes_na_new

    def consensus(self):
        """
        Args:
          M0 (ma.masked_array): votes matrix; rows = voters, columns = Decisions.
          Scales (list): list of dicts for each Decision
            {
              scaled (bool): True if scalar, False if binary (boolean)
              min (float): minimum allowed value (0 if binary)
              max (float): maximum allowed value (1 if binary)
            }

        Returns:
          dict: consensus results

        """
        M0 = self.votes
        Scales = self.decision_bounds
        Rep = self.weights
        CatchP = self.catch_p
        MaxRow = self.max_row
        Verbose = self.verbose

        # Fill the default reputations (egalitarian) if none are provided...
        # unrealistic and only for testing.
        if type(Rep) is int:
            Rep = self.DemocracyCoin(M0)

        # Fill the default scales (binary) if none are provided.
        # In practice, this would also never be used.
        if Scales is None:
            ScaledIndex = [False] * M0.shape[1]
            MScaled = M0
        else:
            ScaledIndex = [scale["scaled"] for scale in Scales]
            MScaled = self.Rescale(M0, Scales)

        # Handle Missing Values
        Filled = self.FillNa(MScaled, ScaledIndex, Rep, CatchP, Verbose)

        # Consensus - Row Players
        # New Consensus Reward
        PlayerInfo = self.GetRewardWeights(Filled, Rep, .1, Verbose)
        AdjLoadings = PlayerInfo['FirstL']

        # Column Players (The Decision Creators)
        # Calculation of Reward for Decision Authors
        # Consensus - "Who won?" Decision Outcome    
        # Simple matrix multiplication ... highest information density at RowBonus,
        # but need DecisionOutcomes.Raw to get to that
        DecisionOutcomes_Raw = np.dot(PlayerInfo['SmoothRep'], Filled).squeeze()

        # Discriminate Based on Contract Type
        ncols = Filled.shape[1]
        for i in range(ncols):
            # Our Current best-guess for this Scaled Decision (weighted median)
            if ScaledIndex[i]:
                DecisionOutcomes_Raw[i] = weighted_median(Filled[:,i],
                                                          PlayerInfo["SmoothRep"].flatten())

        # .5 is obviously undesireable, this function travels from 0 to 1
        # with a minimum at .5
        Certainty = abs(2 * (DecisionOutcomes_Raw - .5))

        # Grading Authors on a curve.
        ConReward = self.GetWeight(Certainty)

        # How well did beliefs converge?
        Avg_Certainty = np.mean(Certainty)

        # The Outcome Itself
        # Discriminate Based on Contract Type
        DecisionOutcome_Adj = []
        for i, raw in enumerate(DecisionOutcomes_Raw):
            DecisionOutcome_Adj.append(self.Catch(raw, CatchP))
            if ScaledIndex[i]:
                DecisionOutcome_Adj[i] = raw

        DecisionOutcome_Final = []
        for i, raw in enumerate(DecisionOutcomes_Raw):
            DecisionOutcome_Final.append(DecisionOutcome_Adj[i])
            if ScaledIndex[i]:
                DecisionOutcome_Final[i] *= (Scales[i]["max"] - Scales[i]["min"])

        # Participation
        # Information about missing values
        NAmat = M0 * 0
        NAmat[NAmat.mask] = 1  # indicator matrix for missing

        # Participation Within Decisions (Columns)
        # % of reputation that answered each Decision
        ParticipationC = 1 - np.dot(PlayerInfo['SmoothRep'], NAmat)

        # Participation Within Agents (Rows)
        # Many options
        # 1- Democracy Option - all Decisions treated equally.
        ParticipationR = 1 - NAmat.sum(axis=1) / NAmat.shape[1]

        # General Participation
        PercentNA = 1 - np.mean(ParticipationC)

        # Possibly integrate two functions of participation? Chicken and egg problem...
        if Verbose:
            print('*Participation Information*')
            print('Voter Turnout by question')
            print(ParticipationC)
            print('Voter Turnout across questions')
            print(ParticipationR)

        # Combine Information
        # Row
        NAbonusR = self.GetWeight(ParticipationR)
        RowBonus = NAbonusR * PercentNA + PlayerInfo['SmoothRep'] * (1 - PercentNA)

        # Column
        NAbonusC = self.GetWeight(ParticipationC)
        ColBonus = NAbonusC * PercentNA + ConReward * (1 - PercentNA)

        Output = {
            'Original': M0.base,
            'Filled': Filled.base,
            'Agents': {
                'OldRep': PlayerInfo['OldRep'][0],
                'ThisRep': PlayerInfo['ThisRep'][0],
                'SmoothRep': PlayerInfo['SmoothRep'][0],
                'NArow': NAmat.sum(axis=1).base,
                'ParticipationR': ParticipationR.base,
                'RelativePart': NAbonusR.base,
                'RowBonus': RowBonus.base,
                },
            'Decisions': {
                'First Loading': AdjLoadings,
                'DecisionOutcomes_Raw': DecisionOutcomes_Raw,
                'Consensus Reward': ConReward,
                'Certainty': Certainty,
                'NAs Filled': NAmat.sum(axis=0),
                'ParticipationC': ParticipationC,
                'Author Bonus': ColBonus,
                'DecisionOutcome_Final': DecisionOutcome_Final,
                },
            'Participation': 1 - PercentNA,
            'Certainty': Avg_Certainty,
        }
        return Output


def main(argv=None):
    if argv is None:
        argv = sys.argv
    try:
        short_opts = 'h'
        long_opts = ['help']
        opts, vals = getopt.getopt(argv[1:], short_opts, long_opts)
    except getopt.GetoptError as e:
        sys.stderr.write(e.msg)
        sys.stderr.write("for help use --help")
        return 2
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(__doc__)
            return 0

if __name__ == '__main__':
    sys.exit(main())
