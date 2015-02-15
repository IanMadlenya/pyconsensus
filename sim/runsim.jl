include("simulate.jl")

@pyimport pyconsensus

liar_threshold_range = 0.1:0.05:0.95

# Empirically, a 90% variance threshold seems to work best!
# (only used for fixed_threshold and fixed_threshold_sum)
# variance_threshold_range = 0.5:0.05:0.95
# sim_data = sensitivity(liar_threshold_range, variance_threshold_range)

sim_data = sensitivity(liar_threshold_range)

# include("pyplot_plots.jl")
include("gadfly_plots.jl")
