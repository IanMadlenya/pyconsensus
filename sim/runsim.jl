using DataFrames

include("simulate.jl")

@pyimport pyconsensus

liar_threshold_range = 0.1:0.1:0.9

println("Simulating...")

sim_data = sensitivity(liar_threshold_range)

println("Building plots...")

include("plots.jl")
