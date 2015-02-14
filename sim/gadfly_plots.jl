using Gadfly

# Comparison to single-component implementation
function heatmaps(algo)
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

target = last(findmax(sum(sim_data["exp_correct"] - sim_data["ref_correct"], 2)))
# target = find(sim_data["variance_threshold"] .== 0.75)

# Plot vtrue values vs liar_threshold parameter
ref_errbars = (sim_data["ref_vtrue"] - sim_data["ref_vtrue_std"],
               sim_data["ref_vtrue"] + sim_data["ref_vtrue_std"])
exp_errbars = (sim_data["exp_vtrue"] - sim_data["exp_vtrue_std"],
               sim_data["exp_vtrue"] + sim_data["exp_vtrue_std"])
pl_vtrue = plot(layer(x=sim_data["liar_threshold"], y=sim_data["ref_vtrue"][:,target],
                      ymin=ref_errbars[1], ymax=ref_errbars[2], 
                      Geom.line, Geom.errorbar, color=["single component"]),
                layer(x=sim_data["liar_threshold"], y=sim_data["exp_vtrue"][:,target],
                      ymin=exp_errbars[1], ymax=exp_errbars[2], 
                      Geom.line, color=["multiple component"]),
                Guide.XLabel("fraction liars"), Guide.YLabel("relative reward"))
pl_vtrue_file = "sens_vtrue_$algo.svg"
draw(SVG(pl_vtrue_file, 12inch, 6inch), pl_vtrue)

# Plot beats values vs liar_threshold parameter
ref_errbars = (sim_data["ref_beats"] - sim_data["ref_beats_std"],
               sim_data["ref_beats"] + sim_data["ref_beats_std"])
exp_errbars = (sim_data["exp_beats"] - sim_data["exp_beats_std"],
               sim_data["exp_beats"] + sim_data["exp_beats_std"])
pl_beats = plot(layer(x=sim_data["liar_threshold"], y=sim_data["ref_beats"][:,target],
                      ymin=ref_errbars[1], ymax=ref_errbars[2], 
                      Geom.line, color=["single component"]),
                layer(x=sim_data["liar_threshold"], y=sim_data["exp_beats"][:,target],
                      ymin=exp_errbars[1], ymax=exp_errbars[2], 
                      Geom.line, color=["multiple component"]),
                Guide.XLabel("fraction liars"), Guide.YLabel("beats"))
pl_beats_file = "sens_beats_$algo.svg"
draw(SVG(pl_beats_file, 12inch, 6inch), pl_beats)

# Plot % correct vs liar_threshold parameter
ref_errbars = (sim_data["ref_correct"] - sim_data["ref_correct_std"],
               sim_data["ref_correct"] + sim_data["ref_correct_std"])
exp_errbars = (sim_data["exp_correct"] - sim_data["exp_correct_std"],
               sim_data["exp_correct"] + sim_data["exp_correct_std"])
pl_correct = plot(layer(x=sim_data["liar_threshold"], y=sim_data["ref_correct"][:,target],
                        ymin=ref_errbars[1], ymax=ref_errbars[2], 
                        Geom.line, color=["single component"]),
                  layer(x=sim_data["liar_threshold"], y=sim_data["exp_correct"][:,target],
                        ymin=exp_errbars[1], ymax=exp_errbars[2], 
                        Geom.line, color=["multiple component"]),
                  Guide.XLabel("fraction liars"), Guide.YLabel("percent correct answers"))
pl_correct_file = "sens_correct_$algo.svg"
draw(SVG(pl_correct_file, 12inch, 6inch), pl_correct)

# heatmaps(sim_data["algo"])
