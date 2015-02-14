using Gadfly

# Heatmaps
function heatmaps(algo)

    # Comparison to single-component implementation
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

algo = sim_data["algo"]
target = last(findmax(sum(sim_data["exp_correct"] - sim_data["ref_correct"], 1)))
num_metrics = 2
gridrows = length(sim_data["liar_threshold"])

liar_threshold = repmat(sim_data["liar_threshold"], 2*num_metrics, 1)[:] * 100

data = [sim_data["ref_beats"][:,target];
        # sim_data["ref_vtrue"][:,target];
        sim_data["ref_correct"][:,target];
        sim_data["exp_beats"][:,target];
        # sim_data["exp_vtrue"][:,target];
        sim_data["exp_correct"][:,target]]

algos = [fill!(Array(String, int(length(data)/2)), "reference");
         fill!(Array(String, int(length(data)/2)), "experimental")]

metrics = repmat([fill!(Array(String, gridrows), "% beats");
                  # fill!(Array(String, gridrows), "liars' reward");
                  fill!(Array(String, gridrows), "% correct")], 2, 1)[:]

error_minus = [
    sim_data["ref_beats"][:,target] - sim_data["ref_beats_std"][:,target],
    # sim_data["ref_vtrue"][:,target] - sim_data["ref_vtrue_std"][:,target],
    sim_data["ref_correct"][:,target] - sim_data["ref_correct_std"][:,target],
    sim_data["exp_beats"][:,target] - sim_data["exp_beats_std"][:,target],
    # sim_data["exp_vtrue"][:,target] - sim_data["exp_vtrue_std"][:,target],
    sim_data["exp_correct"][:,target] - sim_data["exp_correct_std"][:,target],
]
error_plus = [
    sim_data["ref_beats"][:,target] + sim_data["ref_beats_std"][:,target],
    # sim_data["ref_vtrue"][:,target] + sim_data["ref_vtrue_std"][:,target],
    sim_data["ref_correct"][:,target] + sim_data["ref_correct_std"][:,target],
    sim_data["exp_beats"][:,target] + sim_data["exp_beats_std"][:,target],
    # sim_data["exp_vtrue"][:,target] + sim_data["exp_vtrue_std"][:,target],
    sim_data["exp_correct"][:,target] + sim_data["exp_correct_std"][:,target],
]

df = DataFrame(metric=metrics,
               liar_threshold=liar_threshold,
               data=data,
               error_minus=error_minus,
               error_plus=error_plus,
               algorithm=algos)

# Plot metrics vs liar_threshold parameter
set_default_plot_size(12inch, 6inch)
pl = plot(df,
          x=:liar_threshold, y=:data,
          ymin=:error_minus, ymax=:error_plus,
          color=:algorithm, ygroup=:metric,
          Guide.XLabel("% liars"), Guide.YLabel(""),
          Geom.subplot_grid(Geom.point, Geom.line, Geom.errorbar,
                            Guide.xticks(ticks=liar_threshold, label=false),
                            free_y_axis=false))
pl_file = "sens_$algo.svg"
draw(SVG(pl_file, 12inch, 6inch), pl)

# heatmaps(sim_data["algo"])
