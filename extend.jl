using PyCall
using DataFrames
using StatsBase
using JointMoments
using MultivariateStats

@pyimport pyconsensus

extension = (length(ARGS) > 0) ? ARGS[1] : "sim"

function oracle_results(A)
    println(string("ICA convergence: ", A["ica_convergence"]))
    if A["ica_convergence"]
        old_rep = A["agents"]["old_rep"]        # previous reputation
        this_rep = A["agents"]["this_rep"]      # from this round
        smooth_rep = A["agents"]["smooth_rep"]  # weighted sum

        if extension == "sim"
            vtrue = this_rep - this_rep[first(find(players .== "true"))]
            df2 = convert(DataFrame, [players vtrue this_rep smooth_rep])
            colnames2 = names(df2)
            colnames2[1] = "player"
        else
            df2 = convert(DataFrame, [old_rep this_rep smooth_rep])
            colnames2 = names(df2)
            colnames2[1] = "old_rep"
        end

        colnames2[2] = "vs true"
        colnames2[3] = "this_rep"
        colnames2[4] = "smooth_rep"
        names!(df2, colnames2)

        display(df2)
        println()

        vtrue
    end
end

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

# Random test case
elseif extension == "random"
    num_reports = 100
    num_events = 100
    reports = convert(Array{Float64,2}, rand(-1:2:1, num_reports, num_events))
    reports[reports .== 0] = NaN
    reputation = rand(1:100, num_reports)

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

elseif extension == "sim"
    # 1. Generate artificial "true, distort (semi-true), liar" list
    COLLUDE = 0.5     # 0.6 = 60% chance that liars' lies will be identical
    DISTORT = 0.25    # 0.25 = 25% chance of random incorrect answer
    num_events = 10
    num_players = 30

    honesty = rand(num_players)
    players = fill("", num_players)
    players[honesty .>= 0.5] = "true"
    players[0.25 .< honesty .< 0.5] = "distort"
    players[honesty .<= 0.25] = "liar"

    # 2. Build report matrix from this list
    trues = find(players .== "true")
    distorts = find(players .== "distort")
    liars = find(players .== "liar")
    num_trues = length(trues)
    num_distorts = length(distorts)
    num_liars = length(liars)

    correct_answers = rand(-1:1, num_events)

    # True: always report correct answer
    reports = zeros(num_players, num_events)
    reports[trues,:] = convert(Array{Float64,2}, repmat(correct_answers', num_trues))

    # Distort: sometimes report incorrect answers at random
    distmask = rand(num_distorts, num_events) .< DISTORT
    correct = convert(Array{Float64,2}, repmat(correct_answers', num_distorts))
    randomized = convert(Array{Float64,2}, rand(-1:1, num_distorts, num_events))
    reports[distorts,:] = correct.*~distmask + randomized.*distmask

    # Liar: report answers at random (but with a high chance
    #       of being equal to other liars' answers)
    reports[liars,:] = convert(Array{Float64,2}, rand(-1:1, num_liars, num_events))

    # Collusion
    for i in 1:num_liars-1
        # Pairs
        diceroll = first(rand(1))
        if diceroll < COLLUDE
            # println("pair")
            reports[liars[i],:] = reports[liars[i+1],:]
        end
        # Triples
        if i + 2 < num_liars
            # println("triple")
            if diceroll < COLLUDE^2
                reports[liars[i],:] = reports[liars[i+2],:]
            end
        end
    end

    # 3. Optimize RMSD between actual this_rep dispensed, and an ideal this_rep
    display([players reports])
    println()

    reputation = ones(num_players)

end

println("With ICA")
oracle = pyconsensus.Oracle(reports=reports, reputation=reputation, run_ica=true)
A = oracle[:consensus]()
# display(convert(DataFrame, A["agents"]))
# display(convert(DataFrame, A["events"]))
# println()
vtrue = oracle_results(A)
if vtrue != nothing
    println(string("vs true sum: ", sum(vtrue)))
end

println("Without ICA")
oracle = pyconsensus.Oracle(reports=reports, reputation=reputation, run_ica=false)
A = oracle[:consensus]()
vtrue = oracle_results(A)
if vtrue != nothing
    println(string("vs true sum: ", sum(vtrue)))
end
