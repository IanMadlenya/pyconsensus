using PyCall
using DataFrames
using Gadfly
using PyPlot
using Dates
using HDF5, JLD

include("simulate.jl")

@pyimport pyconsensus

VERBOSE = false
ITERMAX = 50
num_events = 50
num_players = 100

function imshow(x, colvals, rowvals,
                title::String="", units::String="",
                xlabel::String="", ylabel::String="", args...)
  is, js, values = findnz(x)
  m, n = size(x)
  df = DataFrames.DataFrame(i=rowvals[is], j=colvals[js], value=values)
  plot(df, x="j", y="i", color="value",
         Coord.cartesian(yflip=false, fixed=true)
       , Geom.rectbin, Stat.identity
       # , Scale.x_continuous(minvalue=0.5, maxvalue=n+0.5)
       # , Scale.y_continuous(minvalue=0.5, maxvalue=m+0.5)
       , Guide.title(title) , Guide.colorkey(units)
       , Guide.XLabel(xlabel), Guide.YLabel(ylabel)
       , Theme(panel_fill=color("black"), grid_line_width=0inch)
       , args...)
end

function jldload(fname="sim_2015-02-12T11:50:17.jld")
    file = jldopen(fname, "r")
    variance_threshold = read(file, "variance_threshold")
    percent_liars = read(file, "percent_liars")
    ref_vtrue_median = read(file, "ref_vtrue_median")
    exp_vtrue_median = read(file, "exp_vtrue_median")
    ref_beats_median = read(file, "ref_beats_median")
    exp_beats_median = read(file, "exp_beats_median")
    ref_correct_median = read(file, "ref_correct_median")
    exp_correct_median = read(file, "exp_correct_median")
    close(file)
end

########################
# Sensitivity analysis #
########################

algo = "fixed_threshold"

# Collusion parameter:
# 0.6 = 60% chance that liars' lies will be identical
collude = 0.5

liar_threshold_range = 0.1:0.05:0.9
variance_threshold_range = 0.1:0.05:0.9
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

sim_data = {
    "variance_threshold" => convert(Array, variance_threshold_range),
    "liar_threshold" => convert(Array, liar_threshold_range),
    "ref_vtrue_median" => ref_vtrue_median,
    "exp_vtrue_median" => exp_vtrue_median,
    "ref_beats_median" => ref_beats_median,
    "exp_beats_median" => exp_beats_median,
    "ref_correct_median" => ref_correct_median,
    "exp_correct_median" => exp_correct_median,
}

jldopen("sim_" * repr(now()) * ".jld", "w") do file
    write(file, "sim_data", sim_data)
end

############
# Heatmaps #
############

xgrid = sim_data["variance_threshold"]
ygrid = sim_data["liar_threshold"]
z = sim_data["exp_correct_median"]

# Surface plot
fig = figure("pyplot_surfaceplot", figsize=(10,10))
ax = fig[:add_subplot](2,1,1, projection = "3d") 
ax[:plot_surface](xgrid, ygrid, z,
                  rstride=2, edgecolors="k", cstride=2,
                  cmap=ColorMap("gray"), alpha=0.8, linewidth=0.25) 
xlabel("variance threshold") 
ylabel("liar threshold")
title("")

# Comparison to single-component implementation
draw(
    SVG(string("compare_heatmap_vtrue_", algo, ".svg"),
        12inch, 12inch),
    imshow(sim_data["ref_vtrue_median"] - sim_data["exp_vtrue_median"],
           sim_data["variance_threshold"],
           sim_data["liar_threshold"],
           "vs true",
           "reward",
           "variance threshold",
           "liar threshold"),
)
draw(
    SVG(string("compare_heatmap_beats_", algo, ".svg"),
        12inch, 12inch),
    imshow(sim_data["ref_beats_median"] - sim_data["exp_beats_median"],
           sim_data["variance_threshold"],
           sim_data["liar_threshold"],
           "liars that escaped punishment (positive = improvement vs reference)",
           "# beats",
           "variance threshold",
           "liar threshold"),
)
draw(
    SVG(string("compare_heatmap_correct_", algo, ".svg"),
        12inch, 12inch),
    imshow(sim_data["ref_correct_median"] - sim_data["exp_correct_median"],
           sim_data["variance_threshold"],
           sim_data["liar_threshold"],
           "event outcomes (negative = improvement vs reference)",
           "% correct",
           "variance threshold",
           "liar threshold"),
)

# Paired sensitivity analysis
draw(
    SVG(string("heatmap_vtrue_", algo, ".svg"),
        12inch, 12inch),
    imshow(sim_data["exp_vtrue_median"],
           sim_data["variance_threshold"],
           sim_data["liar_threshold"],
           "vs true",
           "reward",
           "variance threshold",
           "liar threshold"),
)
draw(
    SVG(string("heatmap_beats_", algo, ".svg"),
        12inch, 12inch),
    imshow(sim_data["exp_beats_median"],
           sim_data["variance_threshold"],
           sim_data["liar_threshold"],
           "liars that escaped punishment",
           "# beats",
           "variance threshold",
           "liar threshold"),
)
draw(
    SVG(string("heatmap_correct_", algo, ".svg"),
        12inch, 12inch),
    imshow(sim_data["exp_correct_median"],
           sim_data["variance_threshold"],
           sim_data["liar_threshold"],
           "event outcomes",
           "% correct",
           "variance threshold",
           "liar threshold"),
)
