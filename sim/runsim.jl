using DataFrames

include("simulate.jl")

@pyimport pyconsensus

liar_threshold_range = 0.1:0.1:0.9

sim_data = sensitivity(liar_threshold_range)

include("plots.jl")
