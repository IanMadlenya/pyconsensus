using PyCall
using DataFrames
using Dates
using Debug
using HDF5, JLD

@pyimport pyconsensus

EVENTS = 500
REPORTERS = 1000
ITERMAX = 100

# Empirically, 90% variance threshold seems best
# for fixed_threshold, 75% for length_threshold
VARIANCE = 0.9
DISTORT = 0

# Collusion: 0.2 => 20% chance liar will copy another liar
# (todo: make this % chance to copy any user, not just liars)
COLLUDE = 0.2
VERBOSE = false
CONSPIRACY = false
ALLWRONG = false
ALGOS = [
    "single_component",
    "fixed_threshold",
    "ica",
    "ica_inverse_scores",
    # "ica_prewhitened",
    # "inverse_scores",
    # "length_threshold",
]
METRICS = [
    "beats",
    "liars_bonus",
    "correct",
]

# "this_rep" is the reputation awarded this round (before smoothing)
function compute_metrics(data, outcomes, this_rep)
    liars_bonus = this_rep - this_rep[first(find(data[:reporters] .== "true"))]
    correct = outcomes .== data[:correct_answers]
    [
        # "liars_bonus": bonus reward liars received (in excess of true reporters')
        :liars_bonus => sum(liars_bonus),

        # "beats" are liars that escaped punishment
        :beats => sum(liars_bonus[data[:liars]] .> 0) / data[:num_liars] * 100,

        # Outcomes that matched our known correct answers list
        :correct => countnz(correct) / EVENTS * 100,
    ]
end

function generate_data(collusion, liar_threshold; variance_threshold=VARIANCE)

    # simplest version: no distortion
    distort_threshold = liar_threshold

    # 1. Generate artificial "true, distort, liar" list
    honesty = rand(REPORTERS)
    reporters = fill("", REPORTERS)
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
        honesty = rand(REPORTERS)
        reporters = fill("", REPORTERS)
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

    correct_answers = rand(-1:1, EVENTS)

    # True: always report correct answer
    reports = zeros(REPORTERS, EVENTS)
    reports[trues,:] = convert(Array{Float64,2}, repmat(correct_answers', num_trues))

    # Distort: sometimes report incorrect answers at random
    distmask = rand(num_distorts, EVENTS) .< DISTORT
    correct = convert(Array{Float64,2}, repmat(correct_answers', num_distorts))
    randomized = convert(Array{Float64,2}, rand(-1:1, num_distorts, EVENTS))
    reports[distorts,:] = correct.*~distmask + randomized.*distmask

    # Liar: report answers at random (but with a high chance
    #       of being equal to other liars' answers)
    reports[liars,:] = convert(Array{Float64,2}, rand(-1:1, num_liars, EVENTS))

    # Alternate: liars always answer incorrectly
    if ALLWRONG
        for i = 1:num_liars
            for j = 1:EVENTS
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
        :reputation => ones(REPORTERS),
        :reporters => reporters,
        :correct_answers => correct_answers,
        :trues => trues,
        :distorts => distorts,
        :liars => liars,
        :num_trues => num_trues,
        :num_distorts => num_distorts,
        :num_liars => num_liars,
        :honesty => honesty,
    ]
end

function simulate(liar_threshold;
                  variance_threshold=VARIANCE,
                  collusion=COLLUDE)
    iterate = (Int64)[]
    i = 1
    reporters = []
    B = Dict()
    for algo in ALGOS
        B[algo] = Dict()
        B[algo]["liars_bonus"] = (Float64)[]
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
            metrics = Dict()
            while ~A[algo]["convergence"]
                A[algo] = pyconsensus.Oracle(
                    reports=data[:reports],
                    reputation=data[:reputation],
                    alpha=1.0,
                    variance_threshold=variance_threshold,
                    algorithm=algo,
                )[:consensus]()
                metrics = compute_metrics(
                    data,
                    A[algo]["events"]["outcomes_final"],
                    A[algo]["agents"]["this_rep"],
                )
            end
            push!(B[algo]["liars_bonus"], metrics[:liars_bonus])
            push!(B[algo]["beats"], metrics[:beats])
            push!(B[algo]["correct"], )
        end

        push!(iterate, i)
        if VERBOSE
            (i == ITERMAX) || (i % 10 == 0) ? println('.') : print('.')
        end
        i += 1
    end

    N = sqrt(ITERMAX)
    C = { "iterate" => iterate }
    for algo in ALGOS
        C[algo] = [
            "mean" => [
                "liars_bonus" => mean(B[algo]["liars_bonus"]),
                "beats" => mean(B[algo]["beats"]),
                "correct" => mean(B[algo]["correct"]),
            ],
            "stderr" => [
                "liars_bonus" => std(B[algo]["liars_bonus"]) / N,
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
                "liars_bonus" => zeros(gridrows, gridcols),
                "beats" => zeros(gridrows, gridcols),
                "correct" => zeros(gridrows, gridcols),
            ],
            "stderr" => [
                "liars_bonus" => zeros(gridrows, gridcols),
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
                        results[algo][statistic]["liars_bonus"][row,col] = C[algo][statistic]["liars_bonus"]
                        results[algo][statistic]["beats"][row,col] = C[algo][statistic]["beats"]
                        results[algo][statistic]["correct"][row,col] = C[algo][statistic]["correct"]
                    end
                end
            end
        else
            C = simulate(liar_threshold, collusion=COLLUDE)
            results["iterate"] = C["iterate"]
            for algo in ALGOS
                for statistic in ("mean", "stderr")
                    results[algo][statistic]["liars_bonus"][row,1] = C[algo][statistic]["liars_bonus"]
                    results[algo][statistic]["beats"][row,1] = C[algo][statistic]["beats"]
                    results[algo][statistic]["correct"][row,1] = C[algo][statistic]["correct"]
                end
            end
        end
    end

    # Save data to file
    variance_thresholds = (isa(variance_threshold_range, Range)) ?
        convert(Array, variance_threshold_range) : variance_threshold_range
    sim_data = {
        "num_reporters" => REPORTERS,
        "num_events" => EVENTS,
        "collude" => COLLUDE,
        "itermax" => ITERMAX,
        "distort" => DISTORT,
        "conspiracy" => CONSPIRACY,
        "allwrong" => ALLWRONG,
        "algos" => ALGOS,
        "metrics" => METRICS,
        "parametrize" => parametrize,
        "liar_threshold" => convert(Array, liar_threshold_range),
        "variance_threshold" => variance_thresholds,
        "iterate" => results["iterate"],
    }
    for algo in ALGOS
        sim_data[algo] = [
            "liars_bonus" => results[algo]["mean"]["liars_bonus"],
            "beats" => results[algo]["mean"]["beats"],
            "correct" => results[algo]["mean"]["correct"],
            "liars_bonus_std" => results[algo]["stderr"]["liars_bonus"],
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

function jldload(fname="sim_2015-02-15T01:41:57.jld")
    jldopen(fname, "r") do file
        read(file, "sim_data")
    end
end

# Auto load data from REPL
~isinteractive() || (sim_data = jldload("sim_2015-02-15T01:41:57.jld"))
