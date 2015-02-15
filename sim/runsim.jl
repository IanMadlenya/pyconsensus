include("simulate.jl")

@pyimport pyconsensus

liar_threshold_range = 0.05:0.025:0.95
# variance_threshold_range = 0.5:0.05:0.95
# sim_data = sensitivity(liar_threshold_range, variance_threshold_range)

sim_data = sensitivity(liar_threshold_range)

include("gadfly_plots.jl")
