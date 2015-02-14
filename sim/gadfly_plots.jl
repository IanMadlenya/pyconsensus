using Gadfly

target = last(findmax(sum(sim_data["exp_correct"] - sim_data["ref_correct"], 1)))

num_metrics = 3
gridrows = length(sim_data["liar_threshold"])

# Build plotting dataframe
liar_threshold = repmat(sim_data["liar_threshold"], 2*num_metrics, 1)[:] * 100
data = [sim_data["ref_beats"][:,target];
        sim_data["ref_vtrue"][:,target];
        sim_data["ref_correct"][:,target];
        sim_data["exp_beats"][:,target];
        sim_data["exp_vtrue"][:,target];
        sim_data["exp_correct"][:,target]]
algos = [fill!(Array(String, int(length(data)/2)), "reference");
         fill!(Array(String, int(length(data)/2)), "experimental")]
metrics = repmat([fill!(Array(String, gridrows), "% beats");
                  fill!(Array(String, gridrows), "liars' reward");
                  fill!(Array(String, gridrows), "% correct")], 2, 1)[:]
error_minus = [
    sim_data["ref_beats"][:,target] - sim_data["ref_beats_std"][:,target],
    sim_data["ref_vtrue"][:,target] - sim_data["ref_vtrue_std"][:,target],
    sim_data["ref_correct"][:,target] - sim_data["ref_correct_std"][:,target],
    sim_data["exp_beats"][:,target] - sim_data["exp_beats_std"][:,target],
    sim_data["exp_vtrue"][:,target] - sim_data["exp_vtrue_std"][:,target],
    sim_data["exp_correct"][:,target] - sim_data["exp_correct_std"][:,target],
]
error_plus = [
    sim_data["ref_beats"][:,target] + sim_data["ref_beats_std"][:,target],
    sim_data["ref_vtrue"][:,target] + sim_data["ref_vtrue_std"][:,target],
    sim_data["ref_correct"][:,target] + sim_data["ref_correct_std"][:,target],
    sim_data["exp_beats"][:,target] + sim_data["exp_beats_std"][:,target],
    sim_data["exp_vtrue"][:,target] + sim_data["exp_vtrue_std"][:,target],
    sim_data["exp_correct"][:,target] + sim_data["exp_correct_std"][:,target],
]
df = DataFrame(metric=metrics,
               liar_threshold=liar_threshold,
               data=data,
               error_minus=error_minus,
               error_plus=error_plus,
               algorithm=algos)

# Plot metrics vs liar_threshold parameter
set_default_plot_size(12inch, 7inch)
pl = plot(df,
    x=:liar_threshold,
    y=:data,
    ymin=:error_minus,
    ymax=:error_plus,
    ygroup=:metric,
    color=:algorithm,
    Guide.XLabel("% liars"),
    Guide.YLabel(""),
    # Guide.xticks(ticks=liar_threshold, label=true),
    Theme(default_color=color("#000099"), panel_stroke=color("#848484")),
    Scale.y_continuous(format=:plain),
    Geom.subplot_grid(
        Geom.point,
        Geom.line,
        Geom.errorbar,
        free_y_axis=true,
    ),
)
pl_file = "sens_" * sim_data["algo"] * ".svg"
draw(SVG(pl_file, 12inch, 7inch), pl)
