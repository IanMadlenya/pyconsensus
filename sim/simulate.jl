using PyCall
using DataFrames
using Dates
using Debug
using HDF5, JLD

@pyimport pyconsensus

EVENTS = 50
REPORTERS = 100
ITERMAX = 50
SQRTN = sqrt(ITERMAX)

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

function compute_metrics(data::Dict{Symbol,Any},
                         outcomes::Array{Any,1},
                         this_rep::Array{Float64,1})

    # "this_rep" is the reputation awarded this round (before smoothing)
    liars_bonus = this_rep - this_rep[first(find(data[:reporters] .== "true"))]
    [
        # "liars_bonus": bonus reward liars received (in excess of true reporters')
        :liars_bonus => sum(liars_bonus),

        # "beats" are liars that escaped punishment
        :beats => sum(liars_bonus[data[:liars]] .> 0) / data[:num_liars] * 100,

        # Outcomes that matched our known correct answers list
        :correct => countnz(outcomes .== data[:correct_answers]) / EVENTS * 100,
    ]
end

function generate_data(collusion::Real,
                       liar_threshold::Real;
                       variance_threshold::Real=VARIANCE)

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
        @inbounds for i = 1:num_liars
            @inbounds for j = 1:EVENTS
                @inbounds while reports[liars[i],j] == correct_answers[j]
                    reports[liars[i],j] = rand(-1:1)
                end
            end
        end
    end

    # All-or-nothing collusion ("conspiracy")
    if CONSPIRACY
        @inbounds for i = 1:num_liars-1
            diceroll = first(rand(1))
            if diceroll < collusion
                reports[liars[i],:] = reports[liars[1],:]
            end
        end

    # "Ordinary" collusion
    else
        @inbounds for i = 1:num_liars-1

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

function simulate(liar_threshold::Real;
                  variance_threshold::Real=VARIANCE,
                  collusion::Real=COLLUDE)
    iterate = (Int64)[]
    i = 1
    reporters = []
    B = Dict()
    @inbounds for algo in ALGOS
        B[algo] = Dict()
        B[algo]["liars_bonus"] = (Float64)[]
        B[algo]["beats"] = (Float64)[]
        B[algo]["correct"] = (Float64)[]
    end
    @inbounds while i <= ITERMAX
        data = generate_data(
            collusion,
            liar_threshold,
            variance_threshold=variance_threshold,
        )

        # consensus
        A = Dict()
        @inbounds for algo in ALGOS
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
            push!(B[algo]["correct"], metrics[:correct])
        end

        push!(iterate, i)
        if VERBOSE
            (i == ITERMAX) || (i % 10 == 0) ? println('.') : print('.')
        end
        i += 1
    end

    C = { "iterate" => iterate }
    @inbounds for algo in ALGOS
        C[algo] = [
            "mean" => [
                "liars_bonus" => mean(B[algo]["liars_bonus"]),
                "beats" => mean(B[algo]["beats"]),
                "correct" => mean(B[algo]["correct"]),
            ],
            "stderr" => [
                "liars_bonus" => std(B[algo]["liars_bonus"]) / SQRTN,
                "beats" => std(B[algo]["beats"]) / SQRTN,
                "correct" => std(B[algo]["correct"]) / SQRTN,  
            ],
        ]
    end
    return C
end

function sensitivity(liar_threshold_range::Range,
                     variance_threshold_range::Union(Real, Range),
                     parametrize::Bool)
    res = Dict()
    gridrows = length(liar_threshold_range)
    gridcols = length(variance_threshold_range)
    @inbounds for algo in ALGOS
        res[algo] = [
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
    @inbounds for (row, liar_threshold) in enumerate(liar_threshold_range)
        println("liar_threshold: ", liar_threshold)

        # Variance threshold parametrization
        if parametrize
            @inbounds for (col, variance_threshold) in enumerate(variance_threshold_range)
                println("  variance_threshold: ", variance_threshold)
                C = simulate(liar_threshold,
                             variance_threshold=variance_threshold,
                             collusion=COLLUDE)
                @inbounds for algo in ALGOS
                    for s in ("mean", "stderr")
                        @inbounds res[algo][s]["liars_bonus"][row,col] = C[algo][s]["liars_bonus"]
                        @inbounds res[algo][s]["beats"][row,col] = C[algo][s]["beats"]
                        @inbounds res[algo][s]["correct"][row,col] = C[algo][s]["correct"]
                    end
                end
            end
        else
            C = simulate(liar_threshold, collusion=COLLUDE)
            res["iterate"] = C["iterate"]
            @inbounds for algo in ALGOS
                for s in ("mean", "stderr")
                    @inbounds res[algo][s]["liars_bonus"][row,1] = C[algo][s]["liars_bonus"]
                    @inbounds res[algo][s]["beats"][row,1] = C[algo][s]["beats"]
                    @inbounds res[algo][s]["correct"][row,1] = C[algo][s]["correct"]
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
        "iterate" => res["iterate"],
    }
    @inbounds for algo in ALGOS
        sim_data[algo] = [
            "liars_bonus" => res[algo]["mean"]["liars_bonus"],
            "beats" => res[algo]["mean"]["beats"],
            "correct" => res[algo]["mean"]["correct"],
            "liars_bonus_std" => res[algo]["stderr"]["liars_bonus"],
            "beats_std" => res[algo]["stderr"]["beats"],
            "correct_std" => res[algo]["stderr"]["correct"],
        ]
    end
    jldopen("sim_" * repr(now()) * ".jld", "w") do file
        write(file, "sim_data", sim_data)
    end
    return sim_data
end

sensitivity(ltr::Range, vtr::Real) = sensitivity(ltr, vtr, false)
sensitivity(ltr::Range) = sensitivity(ltr, VARIANCE)

function jldload(fname::String="sim_2015-02-15T18:25:59.jld")
    jldopen(fname, "r") do file
        read(file, "sim_data")
    end
end

# Auto load data from REPL
~isinteractive() || (sim_data = jldload("sim_2015-02-15T18:25:59.jld"))
