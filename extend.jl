using PyCall
# using StatsBase
# using JointMoments

@pyimport pyconsensus

# true=1, false=-1, indeterminate=0.5, no response=NaN
# reports = [  1  1 -1 NaN;
#              1 -1 -1  -1;
#              1  1 -1  -1;
#              1  1  1  -1;
#            NaN -1  1   1;
#             -1 -1  1   1]
# reputation = [2; 10; 4; 2; 7; 1]
# reputation = PyArray(PyObject(reputation))

num_reports = 100
num_events = 100

reports = convert(Array{Float64,2}, rand(-1:1, num_reports, num_events))
reports[reports .== 0] = NaN
display(reports)

reputation = rand(1:100, num_reports)
display(reputation)

oracle = pyconsensus.Oracle(reports=reports, reputation=reputation)
answer = oracle[:consensus]()

display(answer)
