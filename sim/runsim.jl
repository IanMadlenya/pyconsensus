using PyCall
using DataFrames
using Gadfly
using Dates
using HDF5, JLD

include("simulate.jl")

@pyimport pyconsensus

VERBOSE = false
ITERMAX = 50
num_events = 50
num_players = 100
algo = "fixed_threshold"
# Collusion parameter:
# 0.6 = 60% chance that liars' lies will be identical
collude = 0.5
liar_threshold_range = 0.1:0.05:0.9
variance_threshold_range = 0.1:0.05:0.9

# Surface plot
function surfaceplot(xgrid, ygrid, z)
    fig = figure("pyplot_surfaceplot", figsize=(10,10))
    ax = fig[:add_subplot](111, projection="3d")
    ax[:plot_surface](xgrid, ygrid, z,
                      rstride=2, edgecolors="k", cstride=2,
                      cmap=ColorMap("gray"), alpha=0.8, linewidth=0.25)
    # ax[:plot_wireframe](xgrid, ygrid, z)
    xlabel("variance threshold")
    ylabel("liar threshold")
    zlabel("% correct")
    title("")
end

# Comparison to single-component implementation
function heatmaps()
    draw(
        SVG("compare_heatmap_vtrue_$algo.svg",
            12inch, 12inch),
        imshow(sim_data["ref_vtrue"] - sim_data["exp_vtrue"],
               sim_data["variance_threshold"],
               sim_data["liar_threshold"],
               "vs true",
               "reward",
               "variance threshold",
               "liar threshold"),
    )
    draw(
        SVG("compare_heatmap_beats_$algo.svg",
            12inch, 12inch),
        imshow(sim_data["ref_beats"] - sim_data["exp_beats"],
               sim_data["variance_threshold"],
               sim_data["liar_threshold"],
               "liars that escaped punishment (positive = improvement vs reference)",
               "# beats",
               "variance threshold",
               "liar threshold"),
    )
    draw(
        SVG("compare_heatmap_correct_$algo.svg",
            12inch, 12inch),
        imshow(sim_data["ref_correct"] - sim_data["exp_correct"],
               sim_data["variance_threshold"],
               sim_data["liar_threshold"],
               "event outcomes (negative = improvement vs reference)",
               "% correct",
               "variance threshold",
               "liar threshold"),
    )

    # Paired sensitivity analysis
    draw(
        SVG("heatmap_vtrue_$algo.svg",
            12inch, 12inch),
        imshow(sim_data["exp_vtrue"],
               sim_data["variance_threshold"],
               sim_data["liar_threshold"],
               "vs true",
               "reward",
               "variance threshold",
               "liar threshold"),
    )
    draw(
        SVG("heatmap_beats_$algo.svg",
            12inch, 12inch),
        imshow(sim_data["exp_beats"],
               sim_data["variance_threshold"],
               sim_data["liar_threshold"],
               "liars that escaped punishment",
               "# beats",
               "variance threshold",
               "liar threshold"),
    )
    draw(
        SVG("heatmap_correct_$algo.svg",
            12inch, 12inch),
        imshow(sim_data["exp_correct"],
               sim_data["variance_threshold"],
               sim_data["liar_threshold"],
               "event outcomes",
               "% correct",
               "variance threshold",
               "liar threshold"),
    )
end

########################
# Sensitivity analysis #
########################

gridrows = length(liar_threshold_range)
gridcols = length(variance_threshold_range)
ref_vtrue_median = zeros(gridcols, gridrows)
exp_vtrue_median = zeros(gridcols, gridrows)
ref_beats_median = zeros(gridcols, gridrows)
exp_beats_median = zeros(gridcols, gridrows)
ref_correct_median = zeros(gridcols, gridrows)
exp_correct_median = zeros(gridcols, gridrows)
difference_median = zeros(gridcols, gridrows)

for (row, liar_threshold) in enumerate(liar_threshold_range)
    println("liar_threshold: ", liar_threshold)
    for (col, variance_threshold) in enumerate(variance_threshold_range)
        println("  variance_threshold: ", variance_threshold)
        ref_vtrue, ref_beats, exp_vtrue, exp_beats, difference, ref_correct, exp_correct = simulate(algo, collude, liar_threshold, variance_threshold)
        ref_vtrue_median[row,col] = ref_vtrue
        ref_beats_median[row,col] = ref_beats
        exp_vtrue_median[row,col] = exp_vtrue
        exp_beats_median[row,col] = exp_beats
        ref_correct_median[row,col] = ref_correct
        exp_correct_median[row,col] = exp_correct
        difference_median[row,col] = difference
    end
end

#####################
# Save data to file #
#####################

sim_data = [
    "variance_threshold" => convert(Array, variance_threshold_range),
    "liar_threshold" => convert(Array, liar_threshold_range),
    "ref_vtrue" => ref_vtrue_median,
    "exp_vtrue" => exp_vtrue_median,
    "ref_beats" => ref_beats_median,
    "exp_beats" => exp_beats_median,
    "ref_correct" => ref_correct_median,
    "exp_correct" => exp_correct_median,
    "algo" => algo,
]

jldopen("sim_" * repr(now()) * ".jld", "w") do file
    write(file, "sim_data", sim_data)
end

# Plot vtrue values vs liar_threshold parameter
pl_vtrue = plot(layer(x=sim_data["liar_threshold"], y=sim_data["ref_vtrue"],
                      Geom.line, color=["single component"]),
                layer(x=sim_data["liar_threshold"], y=sim_data["exp_vtrue"],
                      Geom.line, color=["multiple component"]),
                Guide.XLabel("fraction liars"), Guide.YLabel("relative reward"))
pl_vtrue_file = "sens_vtrue_$algo.svg"
draw(SVG(pl_vtrue_file, 12inch, 6inch), pl_vtrue)

# Plot beats values vs liar_threshold parameter
pl_beats = plot(layer(x=sim_data["liar_threshold"], y=sim_data["ref_beats"],
                      Geom.line, color=["single component"]),
                layer(x=sim_data["liar_threshold"], y=sim_data["exp_beats"],
                      Geom.line, color=["multiple component"]),
                Guide.XLabel("fraction liars"), Guide.YLabel("beats"))
pl_beats_file = "sens_beats_$algo.svg"
draw(SVG(pl_beats_file, 12inch, 6inch), pl_beats)

# Plot % correct vs liar_threshold parameter
pl_correct = plot(layer(x=sim_data["liar_threshold"], y=sim_data["ref_correct"],
                        Geom.line, color=["single component"]),
                  layer(x=sim_data["liar_threshold"], y=sim_data["exp_correct"],
                        Geom.line, color=["multiple component"]),
                  Guide.XLabel("fraction liars"), Guide.YLabel("percent correct answers"))
pl_correct_file = "sens_correct_$algo.svg"
draw(SVG(pl_correct_file, 12inch, 6inch), pl_correct)

############
# Heatmaps #
############

using PyPlot

xgrid = repmat(sim_data["variance_threshold"]', length(sim_data["liar_threshold"]), 1)
ygrid = repmat(sim_data["liar_threshold"], 1, length(sim_data["variance_threshold"]))
z = sim_data["exp_correct"] - sim_data["ref_correct"]

surfaceplot(xgrid, ygrid, z)
# heatmaps()
