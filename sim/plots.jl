using DataFrames
using Gadfly

const num_algos = length(sim_data["algos"])
const num_metrics = length(sim_data["metrics"]) - 1  # -1 for components
const gridrows = length(sim_data["liar_threshold"])

# Build plotting dataframe
const liar_threshold = repmat(sim_data["liar_threshold"],
                              num_algos*num_metrics,
                              1)[:] * 100
data = (Float64)[]
algos = (String)[]
metrics = (String)[]
error_minus = (Float64)[]
error_plus = (Float64)[]
for algo in sim_data["algos"]
    if sim_data["parametrize"] && algo in ("fixed-var-length", "fixed-variance")
        target = last(findmax(sum(sim_data[algo]["correct"], 1)))
    else
        target = 1
    end
    data = [
        data,
        sim_data[algo]["beats"][:,target],
        sim_data[algo]["liars_bonus"][:,target],
        sim_data[algo]["correct"][:,target],
    ]
    metrics = [
        metrics,
        fill!(Array(String, gridrows), "% beats"),
        fill!(Array(String, gridrows), "liars' bonus"),
        fill!(Array(String, gridrows), "% correct"),
    ]
    error_minus = [
        error_minus,
        sim_data[algo]["beats"][:,target] - sim_data[algo]["beats_std"][:,target],
        sim_data[algo]["liars_bonus"][:,target] - sim_data[algo]["liars_bonus_std"][:,target],
        sim_data[algo]["correct"][:,target] - sim_data[algo]["correct_std"][:,target],
    ]
    error_plus = [
        error_plus,
        sim_data[algo]["beats"][:,target] + sim_data[algo]["beats_std"][:,target],
        sim_data[algo]["liars_bonus"][:,target] + sim_data[algo]["liars_bonus_std"][:,target],
        sim_data[algo]["correct"][:,target] + sim_data[algo]["correct_std"][:,target],
    ]
    if algo == "first-component" || algo == "sztorc"
        algo = "Sztorc"
    elseif algo == "fourth-cumulant" || algo == "cokurtosis"
        algo = "Cokurtosis"
    elseif algo == "covariance-ratio" || algo == "covariance"
        algo = "Covariance"
    elseif algo =="fixed-variance"
        algo = "Fixed-variance"
    end
    algos = [
        algos,
        repmat(fill!(Array(String, gridrows), algo), 3, 1)[:],
    ]
end
df = DataFrame(metric=metrics[:],
               liar_threshold=liar_threshold[:],
               data=data[:],
               error_minus=error_minus[:],
               error_plus=error_plus[:],
               algorithm=algos[:])

# Plot metrics vs liar_threshold parameter
# set_default_plot_size(15inch, 7inch)
optstr = ""
for flag in ("conspiracy", "allwrong", "indiscriminate")
    optstr *= (sim_data[flag]) ? " " * uppercase(flag) : ""
end
infoblurb = string(
    sim_data["num_reporters"],
    " users reporting on ",
    sim_data["num_events"],
    " events (",
    sim_data["itermax"],
    " iterations @ γ = ",
    sim_data["collude"],
    ")",
    optstr,
)
pl = plot(df,
    x=:liar_threshold,
    y=:data,
    ymin=:error_minus,
    ymax=:error_plus,
    ygroup=:metric,
    color=:algorithm,
    Guide.XLabel("% liars"),
    Guide.YLabel(""),
    Guide.Title(infoblurb),
    Theme(panel_stroke=color("#848484")),
    Scale.y_continuous(format=:plain),
    Geom.subplot_grid(
        Geom.point,
        Geom.line,
        Geom.errorbar,
        free_y_axis=true,
    ),
)
pl_file = "metrics_" * repr(now()) * ".svg"
draw(SVG(pl_file, 10inch, 12inch), pl)
