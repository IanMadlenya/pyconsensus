using PyCall
using DataFrames
using Dates
using Debug
using HDF5, JLD
using JointMoments

@pyimport pyconsensus

num_events = 50
num_reporters = 100
ITERMAX = 50
VARIANCE = 0.9
DISTORT = 0
VERBOSE = false
CONSPIRACY = false
ALLWRONG = true
ALGOS = [
    "single_component",
    # "fixed_threshold",
    "fixed_threshold_sum",
    # "inverse_scores",
    "ica",
    # "ica_prewhitened",
    "ica_inverse_scores",
]
METRICS = [
    "beats",
    "vtrue",
    "correct",
]
# Collusion parameter:
# 0.6 = 60% chance that liars' lies will be identical
COLLUDE = 0.0

function process_oracle_results(A, reporters)
    this_rep = A["agents"]["this_rep"]  # from this round
    true_idx = first(find(reporters .== "true"))

    # increase/decrease vs true
    vtrue = this_rep - this_rep[true_idx]

    liars = find(reporters .== "liar")
    beats = sum(vtrue[reporters .== "liar"] .> 0) / length(liars) * 100
    (vtrue, beats)
end

function generate_data(collusion, liar_threshold; variance_threshold=VARIANCE)

    # simplest version: no distortion
    distort_threshold = liar_threshold

    # 1. Generate artificial "true, distort, liar" list
    honesty = rand(num_reporters)
    reporters = fill("", num_reporters)
    reporters[honesty .>= distort_threshold] = "true"
    reporters[liar_threshold .< honesty .< distort_threshold] = "distort"
    reporters[honesty .<= liar_threshold] = "liar"

    # 2. Build report matrix from this list
    trues = find(reporters .== "true")
    distorts = find(reporters .== "distort")
    liars = find(reporters .== "liar")
    num_trues = length(trues)
    num_distorts = length(distorts)
    num_liars = length(liars)
    
    while num_trues == 0 || num_liars == 0
        honesty = rand(num_reporters)
        reporters = fill("", num_reporters)
        reporters[honesty .>= distort_threshold] = "true"
        reporters[liar_threshold .< honesty .< distort_threshold] = "distort"
        reporters[honesty .<= liar_threshold] = "liar"
        trues = find(reporters .== "true")
        distorts = find(reporters .== "distort")
        liars = find(reporters .== "liar")
        num_trues = length(trues)
        num_distorts = length(distorts)
        num_liars = length(liars)
    end

    correct_answers = rand(-1:1, num_events)

    # True: always report correct answer
    reports = zeros(num_reporters, num_events)
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
    if ALLWRONG
        for i = 1:num_liars
            for j = 1:num_events
                while reports[liars[i],j] == correct_answers[j]
                    reports[liars[i],j] = rand(-1:1)
                end
            end
        end
    end

    # All-or-nothing collusion ("conspiracy")
    if CONSPIRACY
        for i = 1:num_liars-1
            diceroll = first(rand(1))
            if diceroll < collusion
                reports[liars[i],:] = reports[liars[1],:]
            end
        end

    # "Ordinary" collusion
    else
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
    end
    ~VERBOSE || display([reporters reports])
    [
        :reports => reports,
        :reputation => ones(num_reporters),
        :reporters => reporters,
        :correct_answers => correct_answers,
    ]
end

@debug function simulate(liar_threshold;
                  variance_threshold=VARIANCE,
                  collusion=COLLUDE)
    iterate = (Int64)[]
    i = 1
    reporters = []
    B = Dict()
    for algo in ALGOS
        B[algo] = Dict()
        B[algo]["vtrue"] = (Float64)[]
        B[algo]["beats"] = (Float64)[]
        B[algo]["correct"] = (Float64)[]
    end
    while i <= ITERMAX
        data = generate_data(
            collusion,
            liar_threshold,
            variance_threshold=variance_threshold,
        )

        # consensus
        A = Dict()
        for algo in ALGOS
            A[algo] = { "convergence" => false }
            while ~A[algo]["convergence"]
                A[algo] = pyconsensus.Oracle(
                    reports=data[:reports],
                    reputation=data[:reputation],
                    alpha=1.0,
                    variance_threshold=variance_threshold,
                    algorithm=algo,
                )[:consensus]()

                # "beats" are liars that escaped punishment
                A[algo]["vtrue"], A[algo]["beats"] = process_oracle_results(
                    A[algo],
                    data[:reporters],
                )
                A[algo]["vtrue"] = sum(A[algo]["vtrue"])
            end
            correctness = A[algo]["events"]["outcomes_final"] .== data[:correct_answers]
            push!(B[algo]["vtrue"], A[algo]["vtrue"])
            push!(B[algo]["beats"], A[algo]["beats"])
            push!(B[algo]["correct"], countnz(correctness) / num_events * 100)
        end

        push!(iterate, i)
        if VERBOSE
            (i == ITERMAX) || (i % 10 == 0) ? println('.') : print('.')
        end
        i += 1
    end

    N = sqrt(ITERMAX)
    C = Dict()
    for algo in ALGOS
        C[algo] = [
            "mean" => [
                "vtrue" => mean(B[algo]["vtrue"]),
                "beats" => mean(B[algo]["beats"]),
                "correct" => mean(B[algo]["correct"]),
            ],
            "stderr" => [
                "vtrue" => std(B[algo]["vtrue"]) / N,
                "beats" => std(B[algo]["beats"]) / N,
                "correct" => std(B[algo]["correct"]) / N,  
            ],
        ]
    end
    return C
end

function sensitivity(liar_threshold_range::Range,
                     variance_threshold_range::Union(Range, Real),
                     parametrize::Bool)

    results = Dict()

    gridrows = length(liar_threshold_range)
    gridcols = length(variance_threshold_range)

    for algo in ALGOS
        results[algo] = [
            "mean" => [
                "vtrue" => zeros(gridrows, gridcols),
                "beats" => zeros(gridrows, gridcols),
                "correct" => zeros(gridrows, gridcols),
            ],
            "stderr" => [
                "vtrue" => zeros(gridrows, gridcols),
                "beats" => zeros(gridrows, gridcols),
                "correct" => zeros(gridrows, gridcols),
            ]
        ]
    end

    for (row, liar_threshold) in enumerate(liar_threshold_range)
        println("liar_threshold: ", liar_threshold)
        if parametrize
            # Variance threshold parametrization
            for (col, variance_threshold) in enumerate(variance_threshold_range)
                println("  variance_threshold: ", variance_threshold)
                C = simulate(liar_threshold,
                             variance_threshold=variance_threshold,
                             collusion=COLLUDE)
                for algo in ALGOS
                    for statistic in ("mean", "stderr")
                        results[algo][statistic]["vtrue"][row,col] = C[algo][statistic]["vtrue"]
                        results[algo][statistic]["beats"][row,col] = C[algo][statistic]["beats"]
                        results[algo][statistic]["correct"][row,col] = C[algo][statistic]["correct"]
                    end
                end
            end
        else
            C = simulate(liar_threshold, collusion=COLLUDE)
            for algo in ALGOS
                for statistic in ("mean", "stderr")
                    results[algo][statistic]["vtrue"][row,1] = C[algo][statistic]["vtrue"]
                    results[algo][statistic]["beats"][row,1] = C[algo][statistic]["beats"]
                    results[algo][statistic]["correct"][row,1] = C[algo][statistic]["correct"]
                end
            end
        end
    end

    #####################
    # Save data to file #
    #####################

    sim_data = {
        "num_reporters" => num_reporters,
        "num_events" => num_events,
        "collude" => COLLUDE,
        "itermax" => ITERMAX,
        "variance_threshold" => convert(Array, variance_threshold_range),
        "liar_threshold" => convert(Array, liar_threshold_range),
    }
    for algo in ALGOS
        sim_data[algo] = [
            "vtrue" => results[algo]["mean"]["vtrue"],
            "beats" => results[algo]["mean"]["beats"],
            "correct" => results[algo]["mean"]["correct"],
            "vtrue_std" => results[algo]["stderr"]["vtrue"],
            "beats_std" => results[algo]["stderr"]["beats"],
            "correct_std" => results[algo]["stderr"]["correct"],
        ]
    end

    jldopen("sim_" * repr(now()) * ".jld", "w") do file
        write(file, "sim_data", sim_data)
    end

    return sim_data
end

sensitivity(ltr::Range, vtr::Real) = sensitivity(ltr, vtr, false)
sensitivity(ltr::Range) = sensitivity(ltr, VARIANCE)

function jldload(fname="sim_2015-02-14T20:37:31.jld")
    jldopen(fname, "r") do file
        read(file, "sim_data")
    end
end

# Auto load data from REPL
~isinteractive() || (sim_data = jldload("sim_2015-02-14T20:37:31.jld"))
