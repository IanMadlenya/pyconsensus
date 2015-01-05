using PyCall
using DataFrames
using StatsBase
using JointMoments
using MultivariateStats

@pyimport pyconsensus

# Default test case
# true=1, false=-1, indeterminate=0.5, no response=NaN
# reports = [  1  1 -1 NaN ;
#              1 -1 -1  -1 ;
#              1  1 -1  -1 ;
#              1  1  1  -1 ;
#            NaN -1  1   1 ;
#             -1 -1  1   1 ]
# reputation = [2; 10; 4; 2; 7; 1]
# reputation = PyArray(PyObject(reputation))

# Random test case
# num_reports = 100
# num_events = 100
# reports = convert(Array{Float64,2}, rand(-1:2:1, num_reports, num_events))
# reports[reports .== 0] = NaN
# display(reports)
# reputation = rand(1:100, num_reports)
# display(reputation)

# Taken from Truthcoin/lib/ConsensusMechanism.r
#           C1.1 C2.1 C3.0 C4.0
# True         1    1    0    0
# Distort 1    1    0    0    0
# True         1    1    0    0
# Distort 2    1    1    1    0
# Liar         0    0    1    1
# Liar         0    0    1    1
reports = [ 1  1  0  0 ;    # True
            1  0  0  0 ;    # Distort 1
            1  1  0  0 ;    # True
            1  1  1  0 ;    # Distort 2
            0  0  1  1 ;    # Liar
            0  0  1  1 ]    # Liar
reports[reports .== 0] = -1
df = convert(DataFrame, reports)
colnames = names(df)
colnames[1] = "C1.1"
colnames[2] = "C2.1"
colnames[3] = "C3.0"
colnames[4] = "C4.0"
names!(df, colnames)

reputation = [1; 1; 1; 1; 1; 1]

oracle = pyconsensus.Oracle(reports=reports, reputation=reputation)
A = oracle[:consensus]()
display(convert(DataFrame, A["agents"]))
display(convert(DataFrame, A["events"]))
# display(A)
