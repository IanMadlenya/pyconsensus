using Gadfly

num_algos = length(ALGOS)
num_metrics = length(METRICS)
gridrows = length(sim_data["liar_threshold"])

# Build plotting dataframe
liar_threshold = repmat(sim_data["liar_threshold"],
                        num_algos*num_metrics,
                        1)[:] * 100
data = (Float64)[]
algos = (String)[]
metrics = (String)[]
error_minus = (Float64)[]
error_plus = (Float64)[]
for algo in ALGOS
    if algo in ("fixed_threshold", "fixed_threshold_sum")
        target = last(findmax(sum(sim_data[algo]["correct"], 1)))
    else
        target = 1
    end
    data = [
        data,
        sim_data[algo]["beats"][:,target],
        sim_data[algo]["vtrue"][:,target],
        sim_data[algo]["correct"][:,target],
    ]
    algos = [
        algos,
        repmat(fill!(Array(String, gridrows), algo), 3, 1)[:],
    ]
    metrics = [
        metrics,
        fill!(Array(String, gridrows), "% beats"),
        fill!(Array(String, gridrows), "liars' reward"),
        fill!(Array(String, gridrows), "% correct"),
    ]
    error_minus = [
        error_minus,
        sim_data[algo]["beats"][:,target] - sim_data[algo]["beats_std"][:,target],
        sim_data[algo]["vtrue"][:,target] - sim_data[algo]["vtrue_std"][:,target],
        sim_data[algo]["correct"][:,target] - sim_data[algo]["correct_std"][:,target],
    ]
    error_plus = [
        error_plus,
        sim_data[algo]["beats"][:,target] + sim_data[algo]["beats_std"][:,target],
        sim_data[algo]["vtrue"][:,target] + sim_data[algo]["vtrue_std"][:,target],
        sim_data[algo]["correct"][:,target] + sim_data[algo]["correct_std"][:,target],
    ]
end
df = DataFrame(metric=metrics[:],
               liar_threshold=liar_threshold[:],
               data=data[:],
               error_minus=error_minus[:],
               error_plus=error_plus[:],
               algorithm=algos[:])

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
    Theme(
        # default_color=color("#000099"),
        panel_stroke=color("#848484"),
    ),
    Scale.y_continuous(format=:plain),
    Geom.subplot_grid(
        Geom.point,
        Geom.line,
        Geom.errorbar,
        free_y_axis=true,
    ),
)
pl_file = "sensitivity.svg"
draw(SVG(pl_file, 12inch, 7inch), pl)
