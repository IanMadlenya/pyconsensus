using DataFrames

@everywhere include("simulate.jl")

LIAR_THRESHOLDS = 0.1:0.1:0.9

function run_simulations(ltr::Range=LIAR_THRESHOLDS)

    # Run parallel simulations
    raw::Array{Dict{String,Any},1} = @sync @parallel (vcat) for liar_threshold in ltr
        println(round(liar_threshold * 100, 2), "% liars")
        simulate(liar_threshold)
    end

    # Set up final results dictionary
    gridrows = length(ltr)
    results = Dict{String,Any}()
    @inbounds for algo in ALGOS
        results[algo] = Dict{String,Dict}()
        @inbounds for s in STATISTICS
            results[algo][s] = Dict{String,Array}()
            @inbounds for m in METRICS
                results[algo][s][m] = zeros(gridrows)
            end
        end
    end

    # Sort results using liar_threshold values
    for (row, liar_threshold) in enumerate(ltr)
        i = 1
        matched = Dict{String,Dict}()
        for i = 1:gridrows
            if raw[i]["liar_threshold"] == liar_threshold
                matched = splice!(raw, i)
                break
            end
        end
        results["iterate"] = matched["iterate"]
        for algo in ALGOS
            for s in STATISTICS
                for m in METRICS
                    results[algo][s][m][row,1] = matched[algo][s][m]
                end
            end
        end
    end
    results
end

function save_data(results::Dict;
                   ltr::Range=LIAR_THRESHOLDS,
                   vtr::Union(Real, Range)=VARIANCE,
                   parametrize::Bool=false)
    # Save data to file
    variance_thresholds = (isa(vtr, Range)) ? convert(Array, vtr) : vtr
    sim_data = (String => Any)[
        "num_reporters" => REPORTERS,
        "num_events" => EVENTS,
        "collude" => COLLUDE,
        "itermax" => ITERMAX,
        "distort" => DISTORT,
        "conspiracy" => CONSPIRACY,
        "allwrong" => ALLWRONG,
        "responses" => RESPONSES,
        "indiscriminate" => INDISCRIMINATE,
        "algos" => ALGOS,
        "metrics" => METRICS,
        "parametrize" => parametrize,
        "liar_threshold" => convert(Array, ltr),
        "variance_threshold" => variance_thresholds,
        "iterate" => results["iterate"],
    ]
    @inbounds for algo in ALGOS
        sim_data[algo] = (String => Array)[
            "liars_bonus" => results[algo]["mean"]["liars_bonus"],
            "beats" => results[algo]["mean"]["beats"],
            "correct" => results[algo]["mean"]["correct"],
            "components" => results[algo]["mean"]["components"],
            "liars_bonus_std" => results[algo]["stderr"]["liars_bonus"],
            "beats_std" => results[algo]["stderr"]["beats"],
            "correct_std" => results[algo]["stderr"]["correct"],
            "components_std" => results[algo]["stderr"]["components"],
        ]
    end
    filename = "data/sim_" * repr(now()) * ".jld"
    jldopen(filename, "w") do file
        write(file, "sim_data", sim_data)
    end
    println("Data saved to ", filename)
    return sim_data
end

println("Running simulations...")

results = run_simulations()
sim_data = save_data(results)

println("Building plots...")

include("plots.jl")
