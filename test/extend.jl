using PyCall
using JointMoments

@pyimport pyconsensus

reports = [ 1  1  0  0 ;
            1  0  0  0 ;
            1  1  0  0 ;
            1  1  1  0 ;
            0  0  1  1 ;
            0  0  1  1 ]

reputation = [1 1 1 1 1 1]

oracle = pyconsensus.Oracle(reports=reports) #, reputation=reputation)

answer = oracle[:consensus]()
