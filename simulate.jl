using PyCall
using DataFrames
using JointMoments

@pyimport pyconsensus

COLLUDE = 0.5     # 0.6 = 60% chance that liars' lies will be identical
DISTORT = 0.25    # 0.25 = 25% chance of random incorrect answer
VERBOSE = false
num_events = 10
num_players = 30

function oracle_results(A)
    this_rep = A["agents"]["this_rep"]      # from this round

    if VERBOSE
        old_rep = A["agents"]["old_rep"]        # previous reputation
        smooth_rep = A["agents"]["smooth_rep"]  # weighted sum
        vtrue = this_rep - this_rep[first(find(players .== "true"))]
        df2 = convert(DataFrame, [players vtrue this_rep smooth_rep])
        colnames2 = names(df2)
        colnames2[1] = "player"
        colnames2[2] = "vs true"
        colnames2[3] = "this_rep"
        colnames2[4] = "smooth_rep"
        names!(df2, colnames2)
        display(df2)
    end

    this_rep - this_rep[first(find(players .== "true"))]
end

function generate_data()

    # 1. Generate artificial "true, distort (semi-true), liar" list
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
            reports[liars[i],:] = reports[liars[i+1],:]
        end
        
        # Triples
        if i + 2 < num_liars
            if diceroll < COLLUDE^2
                reports[liars[i],:] = reports[liars[i+2],:]
            end
        end
    end

    (reports, ones(num_players))
end

function consensus(reports, reputation)

    # With ICA
    A = pyconsensus.Oracle(reports=reports,
                           reputation=reputation,
                           run_fixed_threshold=true)[:consensus]()
    if A["ica_convergence"]
        ica_vtrue = sum(oracle_results(A))

        # Without ICA
        pca_vtrue = oracle_results(
            pyconsensus.Oracle(reports=reports,
                               reputation=reputation)[:consensus]()
        )
        (vtrue == nothing) ? nothing : ica_vtrue - sum(pca_vtrue)
    end
end

function simulate()
    results = (Float64)[]
    i = 0
    while i <= 100
        reports, reputation = generate_data()
        result = consensus(reports, reputation)
        if result != nothing
            push!(results, result)
            i += 1
        end
    end
    println(string("PCA only vs PCA+ICA (", i-1, " iterations; more negative = improvement vs PCA alone):"))
    println(round(mean(results), 6), " +/- ", round(std(results), 6))
end

simulate()
