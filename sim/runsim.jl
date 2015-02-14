using PyCall
using DataFrames
using Dates
using HDF5, JLD

include("simulate.jl")

@pyimport pyconsensus

# Collusion parameter:
# 0.6 = 60% chance that liars' lies will be identical
collude = 0.6
liar_threshold_range = 0.1:0.05:0.9
variance_threshold_range = 0.5:0.05:0.95

########################
# Sensitivity analysis #
########################

algo = "fixed_threshold"

gridrows = length(liar_threshold_range)
gridcols = length(variance_threshold_range)
ref_vtrue_median = zeros(gridcols, gridrows)
exp_vtrue_median = zeros(gridcols, gridrows)
ref_beats_median = zeros(gridcols, gridrows)
exp_beats_median = zeros(gridcols, gridrows)
ref_correct_median = zeros(gridcols, gridrows)
exp_correct_median = zeros(gridcols, gridrows)
difference_median = zeros(gridcols, gridrows)
ref_vtrue_std = zeros(gridcols, gridrows)
exp_vtrue_std = zeros(gridcols, gridrows)
ref_beats_std = zeros(gridcols, gridrows)
exp_beats_std = zeros(gridcols, gridrows)
ref_correct_std = zeros(gridcols, gridrows)
exp_correct_std = zeros(gridcols, gridrows)
difference_std = zeros(gridcols, gridrows)

for (row, liar_threshold) in enumerate(liar_threshold_range)
    println("liar_threshold: ", liar_threshold)
    for (col, variance_threshold) in enumerate(variance_threshold_range)
        println("  variance_threshold: ", variance_threshold)
        medians, stds = simulate(algo, collude, liar_threshold, variance_threshold)
        ref_vtrue_median[row,col] = medians[1]
        ref_beats_median[row,col] = medians[2]
        exp_vtrue_median[row,col] = medians[3]
        exp_beats_median[row,col] = medians[4]
        difference_median[row,col] = medians[5]
        ref_correct_median[row,col] = medians[6]
        exp_correct_median[row,col] = medians[7]
        ref_vtrue_std[row,col] = stds[1]
        ref_beats_std[row,col] = stds[2]
        exp_vtrue_std[row,col] = stds[3]
        exp_beats_std[row,col] = stds[4]
        difference_std[row,col] = stds[5]
        ref_correct_std[row,col] = stds[6]
        exp_correct_std[row,col] = stds[7]
    end
end

#####################
# Save data to file #
#####################

sim_data = [
    "algo" => algo,
    "num_reporters" => num_reporters,
    "num_events" => num_events,
    "collude" => collude,
    "itermax" => ITERMAX,
    "variance_threshold" => convert(Array, variance_threshold_range),
    "liar_threshold" => convert(Array, liar_threshold_range),
    "ref_vtrue" => ref_vtrue_median,
    "exp_vtrue" => exp_vtrue_median,
    "ref_beats" => ref_beats_median,
    "exp_beats" => exp_beats_median,
    "ref_correct" => ref_correct_median,
    "exp_correct" => exp_correct_median,
    "ref_vtrue_std" => ref_vtrue_std,
    "exp_vtrue_std" => exp_vtrue_std,
    "ref_beats_std" => ref_beats_std,
    "exp_beats_std" => exp_beats_std,
    "ref_correct_std" => ref_correct_std,
    "exp_correct_std" => exp_correct_std,
]

jldopen("sim_" * repr(now()) * ".jld", "w") do file
    write(file, "sim_data", sim_data)
end
