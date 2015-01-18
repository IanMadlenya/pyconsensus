using PyCall
using DataFrames
# using JointMoments
# using Winston
using Gadfly

@pyimport pyconsensus

COLLUDE = 0.85    # 0.6 = 60% chance that liars' lies will be identical
DISTORT = 0.0     # 0.2 = 20% chance of random incorrect answer
VERBOSE = false
ITERMAX = 1000
num_events = 50
num_players = 1000

function oracle_results(A, players)
    this_rep = A["agents"]["this_rep"]          # from this round

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

    (reports, ones(num_players), players)
end

function consensus(reports, reputation, players)

    # Experimental (e.g., with ICA)
    A = pyconsensus.Oracle(reports=reports,
                           reputation=reputation,
                           # run_fixed_threshold=true,
                           run_ica=true)[:consensus]()
    if A["convergence"]
        exp_vtrue = sum(oracle_results(A, players))

        # Reference (e.g., without ICA)
        ref_vtrue = sum(oracle_results(
            pyconsensus.Oracle(reports=reports,
                               reputation=reputation)[:consensus](),
            players
        ))
        (ref_vtrue == nothing) ? nothing :
            (ref_vtrue, exp_vtrue, exp_vtrue - ref_vtrue)
    end
end

function simulate()
    ref_vtrue = (Float64)[]
    exp_vtrue = (Float64)[]
    difference = (Float64)[]
    iterate = (Int64)[]
    i = 1
    while i <= ITERMAX
        reports, reputation, players = generate_data()
        result = consensus(reports, reputation, players)
        if result != nothing
            push!(ref_vtrue, result[1])
            push!(exp_vtrue, result[2])
            push!(difference, result[3])
            push!(iterate, i)
            # if VERBOSE
                (i == ITERMAX) || (i % 10 == 0) ? println('.') : print('.')
            # end
            i += 1
        end
    end
    println("Reference:    ",
            round(median(ref_vtrue), 6), " +/- ", round(std(ref_vtrue), 6))
    println("Experimental: ",
            round(median(exp_vtrue), 6), " +/- ", round(std(exp_vtrue), 6))
    println("Reference vs experimental (", i-1,
            " iterations; negative = improvement vs reference):")
    println(round(median(difference), 6), " +/- ", round(std(difference), 6))

    # Plot results (Winston)
    # figure(width=1000, height=600)
    # P = FramedPlot(title="simulated consensus",
    #                xlabel="iteration",
    #                ylabel="Δ reward")
    # add(P, Curve(iterate, ref_vtrue, color="blue"))
    # add(P, Curve(iterate, exp_vtrue, color="red"))
    # add(P, Curve(iterate, difference, color="black"))
    # display(P)
    # file(P, "sim.png")

    # Plot results (Gadfly)
    println("Building plot...")
    pl = plot(layer(x=iterate, y=ref_vtrue,
                    Geom.line, color=["reference"]),
              layer(x=iterate, y=exp_vtrue,
                    Geom.line, color=["experimental"]),
              layer(x=iterate, y=difference,
                    Geom.line, color=["difference"]),
              Guide.XLabel("iteration"), Guide.YLabel("Δ reward"))
    draw(SVG("sim.svg", 12inch, 6inch), pl)
end

simulate()
