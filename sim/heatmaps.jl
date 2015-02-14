using Gadfly

function heatmap(x, colvals, rowvals,
                 title::String="", units::String="",
                 xlabel::String="", ylabel::String="", args...)
  is, js, values = findnz(x)
  m, n = size(x)
  df = DataFrames.DataFrame(i=rowvals[is], j=colvals[js], value=values)
  plot(df, x="j", y="i", color="value",
         Coord.cartesian(yflip=false, fixed=true)
       , Geom.rectbin, Stat.identity
       , Guide.title(title) , Guide.colorkey(units)
       , Guide.XLabel(xlabel), Guide.YLabel(ylabel)
       , Theme(panel_fill=color("black"), grid_line_width=0inch)
       , args...)
end

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

heatmaps(sim_data["algo"])
