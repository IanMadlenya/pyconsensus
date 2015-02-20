include("simulate.jl")

@pyimport pyconsensus

liar_threshold_range = 0.1:0.05:0.9
# variance_threshold_range = 0.5:0.05:0.95
# sim_data = sensitivity(liar_threshold_range, variance_threshold_range)

sim_data = sensitivity(liar_threshold_range)

include("plots.jl")
