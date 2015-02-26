using DataFrames
using Gadfly

const num_algos = length(sim_data["algos"])
const num_metrics = length(sim_data["metrics"]) - 1  # -1 for components
const gridrows = length(sim_data["liar_threshold"])

METRIC = ("beats", "% beats")
# METRIC = ("liars_bonus", "liars' bonus")
# METRIC = ("correct", "% correct")

# Build plotting dataframe
const liar_threshold = repmat(sim_data["liar_threshold"],
                              num_metrics,
                              1)[:] * 100
data = (Float64)[]
algos = (String)[]
metrics = (String)[]
error_minus = (Float64)[]
error_plus = (Float64)[]
for algo in sim_data["algos"]
    data = [
        data,
        sim_data[algo][METRIC[1]][:,1],
    ]
    metrics = [
        metrics,
        fill!(Array(String, gridrows), METRIC[2]),
    ]
    error_minus = [
        error_minus,
        sim_data[algo][METRIC[1]][:,1] - sim_data[algo][METRIC[1] * "_std"][:,1],
    ]
    error_plus = [
        error_plus,
        sim_data[algo][METRIC[1]][:,1] + sim_data[algo][METRIC[1] * "_std"][:,1],
    ]
    if algo == "first-component"
        algo = "Sztorc"
    elseif algo == "fourth-cumulant"
        algo = "Cokurtosis"
    elseif algo == "covariance-ratio"
        algo = "Covariance"
    elseif algo =="fixed-variance"
        algo = "Fixed-variance"
    end
    algos = [
        algos,
        repmat(fill!(Array(String, gridrows), algo), 1, 1)[:],
    ]
end
df = DataFrame(
    metric=metrics[:],
    liar_threshold=liar_threshold[:],
    data=data[:],
    error_minus=error_minus[:],
    error_plus=error_plus[:],
    algorithm=algos[:],
)

# Plot metrics vs liar_threshold parameter
optstr = ""
for flag in ("conspiracy", "allwrong", "indiscriminate")
    optstr *= (sim_data[flag]) ? " $flag" : ""
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
    color=:algorithm,
    Guide.XLabel("% liars"),
    Guide.YLabel(METRIC[2]),
    Guide.Title(infoblurb),
    Theme(panel_stroke=color("#848484")),
    Scale.y_continuous(
        format=:plain,
        minvalue=-0.5,
        maxvalue=maximum(df[:error_plus]),
    ),
    Geom.point,
    Geom.line,
    Geom.errorbar,
)
pl_file = METRIC[1] * "_" * repr(now()) * ".svg"
draw(SVG(pl_file, 10inch, 7inch), pl)
