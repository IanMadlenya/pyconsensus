using PyPlot

# Surface plot
function surfaceplot(xgrid, ygrid, z)
    fig = figure("pyplot_surfaceplot", figsize=(10,10))
    ax = fig[:add_subplot](111, projection="3d")
    ax[:plot_surface](xgrid, ygrid, z,
                      rstride=2, edgecolors="k", cstride=2,
                      cmap=ColorMap("gray"),
                      alpha=0.8, linewidth=0.25)
    # ax[:plot_wireframe](xgrid, ygrid, z)
    xlabel("variance threshold")
    ylabel("liar threshold")
    zlabel("% correct")
    title("")
end

xgrid = repmat(sim_data["variance_threshold"]', length(sim_data["liar_threshold"]), 1)
ygrid = repmat(sim_data["liar_threshold"], 1, length(sim_data["variance_threshold"]))
z = sim_data["exp_correct"] - sim_data["ref_correct"]

# surfaceplot(xgrid, ygrid, z)
