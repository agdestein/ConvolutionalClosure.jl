if isdefined(@__MODULE__, :LanguageServer)
    include("../src/ConvolutionalClosure.jl")
    using .ConvolutionalClosure
end

# # Closing a filtered equation
#
# In this example we learn a non-linear differential closure term for a
# filtered 1D equation.

using ConvolutionalClosure
using JLD2
using LaTeXStrings
using LinearAlgebra
using Lux
using Plots
using Printf
using SciMLSensitivity
using SparseArrays

# Domain length
# l() = 8π
# l() = 2π
l() = 1.0

# Viscosity (if applicable)
μ() = 0.001

# Reference simulation time
tref() = 0.1

## Equation
# equation() = Convection(l())
# equation() = Diffusion(l(), μ())
equation() = Burgers(l(), μ())
# equation() = KortewegDeVries(l())
# equation() = KuramotoSivashinsky(l())
# equation() = Schrodinger()

# Fine discretization
N = 500
y = LinRange(0, l(), N + 1)[2:end]
Δy = l() / N

# Coarse discretization
M = 50
x = LinRange(0, l(), M + 1)[2:end]
Δx = l() / M

# Filter widths
ΔΔ(x) = 5 / 100 * l() * (1 + 1 / 3 * sin(2π * x / l()))
Δ = ΔΔ.(x)
plot(x, Δ; xlabel = "x", legend = false, title = "Filter width Δ(x)")
# ylims!((0, ylims()[2]))

savefig(loc * "filter_width.pdf")

# # Filter width
# Δ = 5Δx

# Discrete filter matrix
W = sum(-1:1) do z
    d = x .- y' .- z .* l()
    gaussian.(Δ, d) .* (abs.(d) .≤ 3 ./ 2 .* Δ)
    # top_hat.(Δ, d)
end
W = W ./ sum(W; dims = 2)
W = sparse(W)
dropzeros!(W)
plotmat(W; title = "W")

savefig(loc * "W.pdf")

## Example solution
u₀ = @. 1 + 0 * y
u₀ = @. sin(2π * y / l()) + sin(2π * 3y / l()) + cos(2π * 5y / l())
u₀ = @. sin(2π * y / l())
u₀ = @. exp(-(y / l() - 0.5)^2 / 0.005)
u₀ = @. 1.0 * (abs(y / l() - 0.5) ≤ 1 / 6)
plot(u₀; xlabel = "x")

# u₀ = complex(u₀)

# Plot example solution
t = LinRange(0, 10 * tref(), 101)
sol = solve_equation(equation(), u₀, nothing, t; reltol = 1e-6, abstol = 1e-8)
Wsol = W * sol
for (i, t) ∈ enumerate(t)
    pl = plot(;
        xlabel = "x",
        title = @sprintf("t = %.2f", t),
        ylims = extrema(real.(sol[:, :])),
    )
    plot!(pl, y, real.(sol[i]); label = "Unfiltered")
    plot!(pl, x, real.(Wsol[:, i]); label = "Filtered")
    display(pl)
    sleep(0.005) # Time for plot pane to update
end

# Momentum
m = [Δy * sum(u) for u ∈ sol.u]
m̄ = [Δx * sum(ū) for ū ∈ eachcol(Wsol)]
plot(t, [m m̄]; xlabel = "t", title = "Momentum", label = ["Unfiltered" "Filtered"])

# Energy
E = [Δy * u'u / 2 for u ∈ sol.u]
Ē = [Δx * ū'ū / 2 for ū ∈ eachcol(Wsol)]
plot(t, [E Ē]; xlabel = "t", title = "Energy", label = ["Unfiltered" "Filtered"])

"""
Linear-ish frequency decay.
"""
decay(k) = 1 / (1 + abs(k))^1.2

## Maximum frequency in initial conditions
K = N ÷ 2
# K = 30

# Number of samples
n_train = 120
n_valid = 10
n_test = 60

# Initial conditions (real-valued)
u₀_train = create_data(y, K, n_train; decay)
u₀_valid = create_data(y, K, n_valid; decay)
u₀_test = create_data(y, K, n_test; decay)
plot(y, u₀_train[:, 1:3]; xlabel = "x", title = "Initial conditions")

# Evaluation times
t_train = LinRange(0, tref(), 41)
t_valid = LinRange(0, 2 * tref(), 26)
t_test = LinRange(0, 10 * tref(), 61)

# Unfiltered solutions
u_train =
    solve_equation(equation(), u₀_train, nothing, t_train; reltol = 1e-4, abstol = 1e-6)
u_valid =
    solve_equation(equation(), u₀_valid, nothing, t_valid; reltol = 1e-4, abstol = 1e-6)
u_test = solve_equation(equation(), u₀_test, nothing, t_test; reltol = 1e-4, abstol = 1e-6)

# Filtered solutions
ū_train = apply_filter(W, u_train)
ū_valid = apply_filter(W, u_valid)
ū_test = apply_filter(W, u_test)

# Filtered time derivatives (for derivative fitting)
dūdt_train = apply_filter(W, equation()(u_train, nothing, 0.0))
dūdt_valid = apply_filter(W, equation()(u_valid, nothing, 0.0))
dūdt_test = apply_filter(W, equation()(u_test, nothing, 0.0))

## Plot some reference solutions
u, ū, t = u_train, ū_train, t_train
# u, ū, t = u_valid, ū_valid, t_valid
# u, ū, t = u_test, ū_test, t_test
iplot = 1:3
for (i, t) ∈ enumerate(t)
    pl = plot(;
        xlabel = "x",
        title = @sprintf("Reference data, t = %.5f", t),
        ylims = extrema(u[:, iplot, :]),
    )
    plot!(pl, y, u[:, iplot, i]; color = 1, label = "Unfiltered")
    plot!(pl, x, ū[:, iplot, i]; color = 2, label = "Filtered exact")
    display(pl)
    sleep(0.05)
end

# Callback for studying convergence
function create_callback(f, ū, t)
    iplot = 1:10
    hist_i = Int[]
    hist_err = zeros(0)
    err_nomodel = relerr(
        Array(
            solve_equation(
                equation(),
                ū[:, iplot, 1],
                nothing,
                t;
                reltol = 1e-4,
                abstol = 1e-6,
            ),
        ),
        ū[:, iplot, :],
    )
    function callback(i, p)
        sol = solve_equation(f, ū[:, iplot, 1], p, t; reltol = 1e-4, abstol = 1e-6)
        err = relerr(Array(sol), ū[:, iplot, :])
        println("Iteration $i \t average relative error $err")
        push!(hist_i, i)
        push!(hist_err, err)
        pl = plot(; title = "Average relative error", xlabel = "Iterations")
        hline!(pl, [err_nomodel]; color = 1, linestyle = :dash, label = "No closure")
        plot!(pl, hist_i, hist_err; label = "With closure")
        display(pl)
    end
end

# Simple model
p₀_simple, simple_closure = convolutional_closure(
    # Kernel radii (nlayer)
    [5],

    # Number of channels (nlayer + 1)
    # First is number of input channels, last must be 1
    [1, 1],

    # Activation functions (nlayer)
    [identity],

    # Bias
    [false];

    # Input channels
    channel_augmenter = u -> hcat(
        # Vanilla channel
        u,

        # # Square channel to mimic non-linear term
        # u .* u,
    ),
)

# Initialize NN
p₀, closure = convolutional_closure(
    # Kernel radii (nlayer)
    [5, 5, 3],

    # Number of channels (nlayer + 1)
    # First is number of input channels, last must be 1
    [2, 4, 3, 1],

    # Activation functions (nlayer)
    [Lux.relu, Lux.relu, identity],

    # Bias
    [true, true, true];

    # Input channels
    channel_augmenter = u -> hcat(
        # Vanilla channel
        u,

        # Square channel to mimic non-linear term
        u .* u,

        # # Filter width channel for non-uniformity (same for each batch)
        # repeat(Δ, 1, 1, size(u, 3)),
    ),
)

"""
Compute right hand side of closed filtered equation.
This is modeled as unfiltered RHS + neural closure term.
"""
filtered_simple(u, p, t) = equation()(u, nothing, t) + simple_closure(u, p, t)
filtered(u, p, t) = equation()(u, nothing, t) + closure(u, p, t)

p_simple_df = train(
    p -> derivative_loss(
        filtered_simple,
        p,

        # Merge solution and time dimension, with new size `(nx, nsolution*ntime)`
        reshape(dūdt_train, M, :),
        reshape(ū_train, M, :);

        # Number of random data samples for each loss evaluation (batch size)
        nuse = 100,

        # Tikhonov regularization weight
        λ = 1e-8,
    ),
    p₀_simple,
    # p_simple_df,
    10_000;
    ncallback = 100,
    callback = create_callback(filtered_simple, ū_valid, t_valid),
)

p_df = train(
    p -> derivative_loss(
        filtered,
        p,

        # Merge solution and time dimension, with new size `(nx, nsolution*ntime)`
        reshape(dūdt_train, M, :),
        reshape(ū_train, M, :);

        # Number of random data samples for each loss evaluation (batch size)
        nuse = 100,

        # Tikhonov regularization weight
        λ = 1e-8,
    ),
    p₀,
    10_000;
    ncallback = 100,
    callback = create_callback(filtered, ū_valid, t_valid),
)

# filename = "output/$(eqname(equation()))_df.jld2"
# jldsave(filename; p_df)
# p_df = load(filename, "p_df")

p_simple_tf = train(
    p -> trajectory_loss(
        filtered_simple,
        p,
        ū_train,
        t_train;

        # Number of initial conditions per evaluation
        nsolution = 5,

        # Number of random time instances per evaluation
        ntime = 20,

        # Tikhonov regularization weight
        λ = 1.0e-8,

        # Tolerances
        reltol = 1e-4,
        abstol = 1e-6,

        ## Sensitivity algorithm for computing gradient
        # sensealg = BacksolveAdjoint(; autojacvec = ZygoteVJP()),
        sensealg = InterpolatingAdjoint(; autojacvec = ZygoteVJP()),
        # sensealg = QuadratureAdjoint(; autojacvec = ZygoteVJP()),
    ),
    # p₀_simple,
    p_simple_df,
    100;
    ncallback = 10,
    callback = create_callback(filtered_simple, ū_valid, t_valid),
)

p_tf = train(
    p -> trajectory_loss(
        filtered,
        p,
        ū_train,
        t_train;

        # Number of initial conditions per evaluation
        nsolution = 5,

        # Number of random time instances per evaluation
        ntime = 20,

        # Tikhonov regularization weight
        λ = 1.0e-8,

        # Tolerances
        reltol = 1e-4,
        abstol = 1e-6,

        ## Sensitivity algorithm for computing gradient
        # sensealg = BacksolveAdjoint(; autojacvec = ZygoteVJP()),
        sensealg = InterpolatingAdjoint(; autojacvec = ZygoteVJP()),
        # sensealg = QuadratureAdjoint(; autojacvec = ZygoteVJP()),
    ),
    # p₀,
    p_df,
    100;
    ncallback = 10,
    callback = create_callback(filtered, ū_valid, t_valid),
)

# filename = "output/$(eqname(equation()))_tf.jld2"
# jldsave(filename; p_tf)
# p_tf = load(filename, "p_tf")

# Example solution
u₀ = @. sin(2π * y / l()) + sin(2π * 3y / l()) + cos(2π * 5y / l())
u₀ = @. sin(2π * y / l())
u₀ = @. exp(-(y / l() - 0.5)^2 / 0.005)
u₀ = @. 1.0 * (abs(y / l() - 0.5) ≤ 1 / 6)
plot(u₀; xlabel = "x")

t = LinRange(0, 10 * tref(), 101)
u = solve_equation(equation(), u₀, nothing, t; reltol = 1e-6, abstol = 1e-8)
ū = W * u

isample = 2
ū, u, t = ū_train[:, isample, :], u_train[:, isample, :], t_train
ū, u, t = ū_valid[:, isample, :], u_valid[:, isample, :], t_valid
ū, u, t = ū_test[:, isample, :], u_test[:, isample, :], t_test

sol_nomodel = solve_equation(equation(), ū[:, 1], nothing, t; reltol = 1e-4, abstol = 1e-6)
sol_simple_df =
    solve_equation(filtered_simple, ū[:, 1], p_simple_df, t; reltol = 1e-4, abstol = 1e-6)
sol_df = solve_equation(filtered, ū[:, 1], p_df, t; reltol = 1e-4, abstol = 1e-6)
# sol_tf = solve_equation(filtered, ū[:, 1], p_tf, t; reltol = 1e-4, abstol = 1e-6)
for (i, t) ∈ enumerate(t)
    pl = plot(;
        xlabel = "x",
        title = @sprintf("Solutions, t = %.3f", t),
        ylims = extrema(ū),
    )
    # plot!(pl, y, u[:, i]; linestyle = :dash, label = "Unfiltered")
    plot!(pl, x, ū[:, i]; label = "Filtered exact")
    plot!(pl, x, sol_nomodel[i]; label = "No closure")
    plot!(pl, x, sol_simple_df[i]; label = "Derivative fit (simple)")
    plot!(pl, x, sol_df[i]; label = "Derivative fit")
    # plot!(pl, x, sol_tf[i]; label = "Trajectory fit")
    display(pl)
    sleep(0.05)
end

# Momentum
m_exact = [Δx * sum(v) for v ∈ eachcol(ū)]
m_nomodel = [Δx * sum(v) for v ∈ sol_nomodel.u]
m_simple_df = [Δx * sum(v) for v ∈ sol_simple_df.u]
m_df = [Δx * sum(v) for v ∈ sol_df.u]
# m_tf = [Δx * sum(v) for v ∈ sol_tf.u]
pl = plot(; xlabel = "t", title = "Filtered Momentum")
plot!(t, m_exact; label = "Exact")
plot!(t, m_nomodel; label = "No closure")
plot!(t, m_simple_df; label = "Derivative fit (simple)")
plot!(t, m_df; label = "Derivative fit")
# plot!(t, m_tf; label = "Trajectory fit")
pl

# Energy
E_exact = [Δx * v'v / 2 for v ∈ eachcol(ū)]
E_nomodel = [Δx * v'v / 2 for v ∈ sol_nomodel.u]
E_simple_df = [Δx * v'v / 2 for v ∈ sol_simple_df.u]
E_df = [Δx * v'v / 2 for v ∈ sol_df.u]
# E_tf = [Δx * v'v / 2 for v ∈ sol_tf.u]
pl = plot(; xlabel = "t", title = "Filtered Energy")
plot!(t, E_exact; label = "Exact")
plot!(t, E_nomodel; label = "No closure")
plot!(t, E_simple_df; label = "Derivative fit (simple)")
plot!(t, E_df; label = "Derivative fit")
# plot!(t, E_tf; label = "Trajectory fit")
pl

plotsol(y, t, u[:, 1, :]; title = "u")
# savefig(loc * "$(eqname(equation()))_u.png")
plotsol(x, t, ū[:, 1, :]; title = "Wu")
# savefig(loc * "$(eqname(equation()))_Wu.png")
plotsol(x, t, sol_df)
# plotsol(x, t, sol_tf[:, 1, :])
plotsol(x, t, sol_nomodel)
plotsol(x, t, (sol_nomodel - ū) ./ norm(ū))
plotsol(x, t, (sol_df - ū) ./ norm(ū))
# plotsol(x, t, (sol_tf - ū) ./ norm(ū))

relerr(Array(sol_nomodel), ū)
relerr(Array(sol_simple_df), ū)
relerr(Array(sol_df), ū)
relerr(Array(sol_tf), ū)

ū, u, t = ū_test, u_test, t_test
v_nomodel =
    solve_equation(equation(), ū[:, :, 1], nothing, t; reltol = 1e-4, abstol = 1e-6)
v_simple_df = solve_equation(
    filtered_simple,
    ū[:, :, 1],
    p_simple_df,
    t;
    reltol = 1e-4,
    abstol = 1e-6,
)
v_df = solve_equation(filtered, ū[:, :, 1], p_df, t; reltol = 1e-4, abstol = 1e-6)
# v_tf = solve_equation(filtered, ū[:, :, 1], p_tf, t; reltol = 1e-4, abstol = 1e-6)

error_momentum(a, b) = [relerr(a[:, :, i], b[:, :, i]) for i ∈ 1:size(a, 3)]
error_total_momentum(a, b) =
    [relerr(sum(a; dims = 1)[:, :, i], sum(b; dims = 1)[:, :, i]) for i ∈ 1:size(a, 3)]
error_energy(a, b) = [relerr(a[:, :, i] .^ 2, b[:, :, i] .^ 2) for i ∈ 1:size(a, 3)]
error_total_energy(a, b) = [
    relerr(sum(abs2, a; dims = 1)[:, :, i], sum(abs2, b; dims = 1)[:, :, i]) for
    i ∈ 1:size(a, 3)
]

errerr(a, b) = error_momentum(a, b)
errerr(a, b) = error_total_momentum(a, b)
errerr(a, b) = error_energy(a, b)
errerr(a, b) = error_total_energy(a, b)

pl = plot(; xlabel = "t", title = "Relative error (test dataset)", legend = :topright)
plot!(t, errerr(v_nomodel, ū); label = "No model")
plot!(t, errerr(v_simple_df, ū); label = "Linear closure")
plot!(t, errerr(v_df, ū); label = "CNN closure")
# plot!(t, err(v_tf, ū); label = "Trajectory fit") pl
pl

savefig(loc * "burgers_relative_errors.pdf")
