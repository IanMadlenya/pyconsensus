using PyCall

@pyimport pyconsensus

reports = [ 1  1  0  0 ;
            1  0  0  0 ;
            1  1  0  0 ;
            1  1  1  0 ;
            0  0  1  1 ;
            0  0  1  1 ]

reputation = [2 10 4 2 7 1]

oracle = pyconsensus.Oracle(reports=reports, reputation=reputation)

answer = oracle[:consensus]()
