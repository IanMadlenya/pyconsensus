using PyCall
using DataFrames
using HDF5, JLD

@pyimport pyconsensus

VERBOSE = false
ITERMAX = 100
num_events = 100
num_players = 500

function oracle_results(A, players)
    this_rep = A["agents"]["this_rep"]          # from this round
    true_idx = first(find(players .== "true"))
    vtrue = this_rep - this_rep[true_idx]
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

function generate_data(collusion, liar_threshold, variance_threshold)

    # simplest version: no distortion
    distort_threshold = liar_threshold

    # 1. Generate artificial "true, distort, liar" list
    honesty = rand(num_players)
    players = fill("", num_players)
    players[honesty .>= distort_threshold] = "true"
    players[liar_threshold .< honesty .< distort_threshold] = "distort"
    players[honesty .<= liar_threshold] = "liar"

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

    # Alternate: liars always answer incorrectly
    # for i = 1:num_liars
    #     for j = 1:num_events
    #         while reports[liars[i],j] == correct_answers[j]
    #             reports[liars[i],j] = rand(-1:1)
    #         end
    #     end
    # end

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

    # # All-or-nothing collusion ("conspiracy")
    # for i = 1:num_liars-1
    #     diceroll = first(rand(1))
    #     if diceroll < collusion
    #         reports[liars[i],:] = reports[liars[1],:]
    #     end
    # end

    ~VERBOSE || display([players reports])

    (reports, ones(num_players), players, correct_answers)
end

function consensus(reports, reputation, players, algo, variance_threshold)

    # Experimental (fixed-threshold)
    if algo == "fixed_threshold"
        A = pyconsensus.Oracle(reports=reports,
                               alpha=1.0,
                               reputation=reputation,
                               variance_threshold=variance_threshold,
                               run_fixed_threshold=true)[:consensus]()
    elseif algo == "inverse_scores"
        A = pyconsensus.Oracle(reports=reports,
                               alpha=1.0,
                               reputation=reputation,
                               run_inverse_scores=true)[:consensus]()
    elseif algo == "ica"
        A = pyconsensus.Oracle(reports=reports,
                               alpha=1.0,
                               reputation=reputation,
                               run_ica=true)[:consensus]()
    elseif algo == "ica_inverse_scores"
        A = pyconsensus.Oracle(reports=reports,
                               alpha=1.0,
                               reputation=reputation,
                               run_ica_inverse_scores=true)[:consensus]()
    elseif algo == "ica_prewhitened"
        A = pyconsensus.Oracle(reports=reports,
                               alpha=1.0,
                               reputation=reputation,
                               run_ica_prewhitened=true)[:consensus]()
    end

    if A["convergence"]
        # "beats" are liars that escaped punishment
        exp_vtrue, exp_beats = oracle_results(A, players)
        exp_vtrue = sum(exp_vtrue)
        exp_outcome_final = A["events"]["outcomes_final"]

        # Reference (single-component)
        ref_A = pyconsensus.Oracle(reports=reports, reputation=reputation, alpha=1.0)[:consensus]()
        ref_vtrue, ref_beats = oracle_results(ref_A, players)
        ref_vtrue = sum(ref_vtrue)
        ref_outcome_final = ref_A["events"]["outcomes_final"]
        ~VERBOSE || display([ref_outcome_final exp_outcome_final ref_outcome_final .== exp_outcome_final])
        (ref_vtrue == nothing) ? nothing :
            (ref_vtrue, exp_vtrue, exp_vtrue - ref_vtrue, ref_beats, exp_beats, ref_outcome_final, exp_outcome_final)
    end
end

function simulate(algo, collusion, liar_threshold, variance_threshold)
    ref_vtrue = (Float64)[]
    exp_vtrue = (Float64)[]
    difference = (Float64)[]
    ref_beats = (Float64)[]
    exp_beats = (Float64)[]
    ref_correct = (Float64)[]
    exp_correct = (Float64)[]
    iterate = (Int64)[]
    i = 1
    players = []
    while i <= ITERMAX
        reports, reputation, players, correct_answers = generate_data(collusion, liar_threshold, variance_threshold)
        while ~("true" in players && "liar" in players)
            reports, reputation, players, correct_answers = generate_data(collusion, liar_threshold, variance_threshold)
        end
        result = consensus(reports, reputation, players, algo, variance_threshold)
        if result != nothing
            ref_correctness = result[6] .== correct_answers
            ref_percent_correct = countnz(ref_correctness) / num_events * 100
            exp_correctness = result[7] .== correct_answers
            exp_percent_correct = countnz(exp_correctness) / num_events * 100
            if VERBOSE && ref_percent_correct != exp_percent_correct
                println("ref_percent_correct: ", ref_percent_correct)
                println("exp_percent_correct: ", exp_percent_correct)
            end
            push!(ref_vtrue, result[1])
            push!(exp_vtrue, result[2])
            push!(difference, result[3])
            push!(ref_beats, result[4])
            push!(exp_beats, result[5])
            push!(ref_correct, ref_percent_correct)
            push!(exp_correct, exp_percent_correct)
            push!(iterate, i)
            if VERBOSE
                (i == ITERMAX) || (i % 10 == 0) ? println('.') : print('.')
            end
            i += 1
        end
    end

    if VERBOSE
        println("Reference:    ",
                round(median(ref_vtrue), 6), " +/- ", round(std(ref_vtrue), 6),
                " (", round(median(ref_beats), 6), " +/- ", round(std(ref_beats), 6), ")")
        println("Experimental: ",
                round(median(exp_vtrue), 6), " +/- ", round(std(exp_vtrue), 6),
                " (", round(median(exp_beats), 6), " +/- ", round(std(exp_beats), 6), ")")
        println("Reference vs experimental (", i-1,
                " iterations; negative = improvement vs reference):")
        println(round(median(difference), 6), " +/- ", round(std(difference), 6))
    end

    map(median, (ref_vtrue, ref_beats, exp_vtrue, exp_beats, difference, ref_correct, exp_correct))
end

function heatmap(x, colvals, rowvals,
                 title::String="", units::String="",
                 xlabel::String="", ylabel::String="", args...)
  is, js, values = findnz(x)
  m, n = size(x)
  df = DataFrames.DataFrame(i=rowvals[is], j=colvals[js], value=values)
  plot(df, x="j", y="i", color="value",
         Coord.cartesian(yflip=false, fixed=true)
       , Geom.rectbin, Stat.identity
       # , Scale.x_continuous(minvalue=0.5, maxvalue=n+0.5)
       # , Scale.y_continuous(minvalue=0.5, maxvalue=m+0.5)
       , Guide.title(title) , Guide.colorkey(units)
       , Guide.XLabel(xlabel), Guide.YLabel(ylabel)
       , Theme(panel_fill=color("black"), grid_line_width=0inch)
       , args...)
end

function jldload(fname="sim_2015-02-12T13:55:51.jld")
    jldopen(fname, "r") do file
        sim_data = read(file, "sim_data")
        return sim_data
    end
end

# Auto load data from REPL
~isinteractive() || (sim_data = jldload())
