dU = reshape(dūdt_train, M, :)
U = reshape(ū_train, M, :)

A = dU * U' / (U * U' + 1e-1 * I)

plotmat(A)

apply_mat(u, p, t) = p * u

function solve_matrix(A, u₀, t; kwargs...)
    problem = ODEProblem(apply_mat, u₀, extrema(t), A)
    solve(problem, Tsit5(); saveat = t, kwargs...)
end


## Plot performance evolution
# ū, u, t = ū_train, u_train, t_train
# ū, u, t = ū_valid, u_valid, t_valid
ū, u, t = ū_test, u_test, t_test
isample = 2
sol = solve_equation(ū[:, [isample], 1], t; reltol = 1e-4, abstol = 1e-6)
sol_NN = solve_filtered(p, ū[:, [isample], 1], t; reltol = 1e-4, abstol = 1e-6)
sol_A = solve_matrix(A, ū[:, isample, 1], t; reltol = 1e-4, abstol = 1e-6)
for (i, t) ∈ enumerate(t)
    pl = plot(;
        xlabel = "x",
        title = @sprintf("Solutions, t = %.3f", t),
        ylims = extrema(u[:, isample, :]),
    )
    # plot!(pl, ξ, u[:, isample, i]; linestyle = :dash, label = "Unfiltered")
    plot!(pl, x, ū[:, isample, i]; label = "Filtered exact")
    # plot!(pl, x, sol[:, 1, i]; label = "No closure")
    plot!(pl, x, sol_NN[:, 1, i]; label = "Neural closure")
    plot!(pl, x, sol_A[:, 1, i]; label = "Fit matrix")
    span = x -> [x - ΔΔ(x) / 2, x + ΔΔ(x) / 2]
    vspan!(pl, span(1 / 4); fillalpha = 0.1, color = 1, label = L"x \pm h(x)");
    vspan!(pl, span(3 / 4); fillalpha = 0.1, color = 1, label = nothing);
    display(pl)
    sleep(0.05)
end


lims = extrema(ū[:, isample, :])
heatmap(t_test, x, ū[:, isample, :]; xlabel = "t", ylabel = "x", clims = lims)
heatmap(t_test, x, sol_NN[:, 1, :]; xlabel = "t", ylabel = "x", clims = lims)
heatmap(t_test, x, sol_A[:, 1, :]; xlabel = "t", ylabel = "x", clims = lims)
heatmap(t_test, x, sol[:, 1, :]; xlabel = "t", ylabel = "x", clims = lims)
heatmap(
    t_test, x,
    (sol_NN[:, 1, :] - ū[:, isample, :]) ./ norm(ū[:, isample, :]);
    xlabel = "t",
    ylabel = "x",
)
heatmap(
    t_test, x,
    (sol_A[:, 1, :] - ū[:, isample, :]) ./ norm(ū[:, isample, :]);
    xlabel = "t",
    ylabel = "x",
)

relerr(sol[:, [1], :], ū[:, [isample], :], t_test)
relerr(sol_NN[:, [1], :], ū[:, [isample], :], t_test)
relerr(sol_A, ū[:, isample, :], t)
