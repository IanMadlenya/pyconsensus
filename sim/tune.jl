using PyCall
using DataFrames
using Gadfly

@pyimport pyconsensus

DISTORT = 0.0     # 0.2 = 20% chance of random incorrect answer
VERBOSE = true
ITERMAX = 100
num_events = 25
num_players = 50

function oracle_results(A, players)
    this_rep = A["agents"]["this_rep"]          # from this round
    vtrue = this_rep - this_rep[first(find(players .== "true"))]

    if VERBOSE
        old_rep = A["agents"]["old_rep"]        # previous reputation
        smooth_rep = A["agents"]["smooth_rep"]  # weighted sum
        df2 = convert(DataFrame, [players vtrue this_rep smooth_rep])
        colnames2 = names(df2)
        colnames2[1] = "player"
        colnames2[2] = "vs true"
        colnames2[3] = "this_rep"
        colnames2[4] = "smooth_rep"
        names!(df2, colnames2)
        display(df2)
    end

    (vtrue, sum(vtrue[players .== "liar"] .> 0))
end

function generate_data(collusion)

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
    for i = 1:num_liars-1

        # Pairs
        diceroll = first(rand(1))
        if diceroll < collusion
            reports[liars[i],:] = reports[liars[i+1],:]

            # Triples
            if i + 2 < num_liars
                if diceroll < collusion^2
                    reports[liars[i],:] = reports[liars[i+2],:]
                end

                # Quadruples
                if i + 3 < num_liars
                    if diceroll < collusion^3
                        reports[liars[i],:] = reports[liars[i+3],:]
                    end
                end
            end
        end
    end

    # All-or-nothing collusion ("conspiracy")
    # for i = 1:num_liars-1
    #     diceroll = first(rand(1))
    #     if diceroll < collusion
    #         reports[liars[i],:] = reports[liars[1],:]
    #     end
    # end

    ~VERBOSE || display([players reports])
    display([players reports])

    (reports, ones(num_players), players, correct_answers)
end

collusion = 0.5
reports, reputation, players, correct_answers = generate_data(collusion)

A = pyconsensus.Oracle(reports=reports, reputation=reputation)[:consensus]()
vtrue, beats = oracle_results(A, players)

correctness = A["events"]["outcome_final"] .== correct_answers'
num_correct = countnz(correctness)
display([A["events"]["outcome_final"] correct_answers' correctness])

println("correct answers: ", num_correct, "/", num_events, " (", num_correct/num_events*100, "%)")
