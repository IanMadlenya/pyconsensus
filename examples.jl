# Default test case
if extension == "default"
    # true=1, false=-1, indeterminate=0.5, no response=NaN
    reports = [  1  1 -1 NaN ;
                 1 -1 -1  -1 ;
                 1  1 -1  -1 ;
                 1  1  1  -1 ;
               NaN -1  1   1 ;
                -1 -1  1   1 ]
    reputation = [2; 10; 4; 2; 7; 1]
    reputation = PyArray(PyObject(reputation))
    consensus(reports, reputation)

# Random test case
elseif extension == "random"
    num_reports = 100
    num_events = 100
    reports = convert(Array{Float64,2}, rand(-1:2:1, num_reports, num_events))
    reports[reports .== 0] = NaN
    reputation = rand(1:100, num_reports)
    consensus(reports, reputation)

elseif extension == "example"
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
    consensus(reports, reputation)
end