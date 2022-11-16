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
using Optimisers
using Plots
using Printf
using Random
using SciMLSensitivity
using Zygote

# Domain length
l() = 8π
# l() = 1.0

# Viscosity (if applicable)
μ() = 0.001

# Reference simulation time
tref() = 0.1

## Equation
# equation() = Convection(l())
# equation() = Diffusion(l(), μ())
# equation() = Burgers(l(), μ())
equation() = KortewegDeVries(l())
# equation() = KuramotoSivashinsky(l())

# Fine discretization
N = 200
ξ = LinRange(0, l(), N + 1)[2:end]
Δξ = l() / N

# Coarse discretization
M = 100
x = LinRange(0, l(), M + 1)[2:end]
Δx = l() / M

# # Filter widths
# ΔΔ(x) = 3Δx * (1 + 1 / 3 * sin(2π * x / l()))
# Δ = ΔΔ.(x)
# plot(x, Δ; xlabel = "x", title = "Filter width")

# Filter width
Δ = 3Δx

# Discrete filter matrix
W = sum(gaussian.(Δ, x .- ξ' .- z .* l()) for z ∈ -2:2)
W = W ./ sum(W; dims = 2)
plotmat(W; title = "Discrete filter")

## Example solution
u₀ = @. sin(2π * ξ / l()) + sin(2π * 3ξ / l()) + cos(2π * 5ξ / l())
u₀ = @. sin(2π * ξ / l())
u₀ = @. exp(-(ξ / l() - 0.5)^2 / 0.005)
plot(u₀; xlabel = "x")

# Plot example solution
t = LinRange(0, 50 * tref(), 101)
sol = solve_equation(equation(), u₀, nothing, t; reltol = 1e-6, abstol = 1e-8)
Wsol = W * sol
for (i, t) ∈ enumerate(t)
    pl = plot(; xlabel = "x", title = @sprintf("t = %.2f", t), ylims = extrema(sol[:, :]))
    plot!(pl, ξ, sol[i]; label = "Unfiltered")
    plot!(pl, x, Wsol[:, i]; label = "Filtered")
    display(pl)
    sleep(0.05) # Time for plot pane to update
end

# Momentum
m = [Δξ * sum(u) for u ∈ sol.u]
m̄ = [Δx * sum(ū) for ū ∈ eachcol(Wsol)]
plot(t, [m m̄]; xlabel = "t", title = "Momentum", label = ["Unfiltered" "Filtered"])

# Energy
E = [Δξ * u'u / 2 for u ∈ sol.u]
Ē = [Δx * ū'ū / 2 for ū ∈ eachcol(Wsol)]
plot(t, [E Ē]; xlabel = "t", title = "Energy", label = ["Unfiltered" "Filtered"])

## Safe (but creates NN with Float32)
# init_weight = Lux.glorot_uniform
# init_bias = Lux.zeros32

# This also works for Float64
init_weight = glorot_uniform_Float64
init_bias = zeros

# Kernel radii (nlayer)
r = [3, 4, 1]

# Number of channels (nlayer + 1)
# First is number of input channels, last must be 1
c = [2, 4, 3, 1]

# Activation functions (nlayer)
a = [Lux.relu, Lux.relu, identity]

# Discrete closure term for filtered equation
NN = Chain(
    # From (nx, nsample) to (nx, nchannel, nsample)
    u -> reshape(u, size(u, 1), 1, size(u, 2)),

    # Create input channels
    u -> hcat(
        # Vanilla channel
        u,

        # Square channel to mimic non-linear term
        u .* u,

        # # Filter width channel for non-uniformity (same for each batch)
        # repeat(Δ, 1, 1, size(u, 3)),
    ),

    # Manual padding along spatial dimension to account for periodicity
    u -> [u[end-sum(r)+1:end, :, :]; u; u[1:sum(r), :, :]],

    # Some convolutional layers to mimic local differential operators
    (Conv((2r[i] + 1,), c[i] => c[i+1], a[i]; init_weight) for i ∈ eachindex(r))...,

    # From (nx, nchannel = 1, nsample) to (nx, nsample)
    u -> reshape(u, size(u, 1), size(u, 3)),

    # # Difference for momentum conservation
    # u -> circshift(u, -1) - circshift(u, 1),
)

# Initialize NN
rng = Random.default_rng()
Random.seed!(rng, 0)
Lux.setup(rng, NN)
params, state = Lux.setup(rng, NN)
p₀, re = Lux.destructure(params)

"""
Compute closure term for given parameters `p`.
"""
closure(u, p, t) = first(Lux.apply(NN, u, re(p), state))

"""
Compute right hand side of closed filtered equation.
This is modeled as unfiltered RHS + neural closure term.
"""
function filtered(u, p, t)
    du = equation()(u, nothing, t)
    c = closure(u, p, t)
    du + c
end

"""
Creating derivative-fitting loss function.
Chooses a random subset (`nuse`) of the data samples at each evaluation.
Note that both `u` and `dudt` are of size `(nx, nsample)`.
"""
function create_loss_derivative_fit(dudt, u; nuse = size(u, 2), λ = 0)
    function loss(p)
        i = Zygote.@ignore sort(shuffle(1:size(u, 2))[1:nuse])
        loss_derivative_fit(filtered, p, dudt[:, i], u[:, i], λ)
    end
    loss
end

"""
Create trajectory-fitting loss function.
Chooses a random subset of the solutions and time points at each evaluation.
Note that `u` is of size `(nx, nsample, ntime)`
"""
function create_loss_trajectory_fit(
    u,
    t;
    nsample = size(u, 2),
    ntime = length(t),
    λ = 0,
    kwargs...,
)
    function loss(p)
        iu = Zygote.@ignore sort(shuffle(1:size(u, 2))[1:nsample])
        it = Zygote.@ignore sort(shuffle(1:length(t))[1:ntime])
        loss_embedded(filtered, p, u[:, iu, it], t[it], λ; kwargs...)
    end
    loss
end

"""
Linear-ish frequency decay.
"""
decay(k) = 1 / (1 + abs(k))

## Maximum frequency in initial conditions
K = N ÷ 2
# K = 30

# Number of samples
n_train = 120
n_valid = 10
n_test = 60

# Initial conditions (real-valued)
u₀_train = create_data(ξ, K, n_train; decay)
u₀_valid = create_data(ξ, K, n_valid; decay)
u₀_test = create_data(ξ, K, n_test; decay)
plot(ξ, u₀_train[:, 1:3]; xlabel = "x", title = "Initial conditions")

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
    plot!(pl, ξ, u[:, iplot, i]; color = 1, label = "Unfiltered")
    plot!(pl, x, ū[:, iplot, i]; color = 2, label = "Filtered exact")
    display(pl)
    sleep(0.05)
end

# Callback for studying convergence
plot_loss = let
    ū, t = ū_valid, t_valid
    # ū, t = ū_test, t_test
    iplot = 1:10
    hist_i = Int[]
    hist_err = zeros(0)
    err_nomodel = relerr(
        solve_equation(
            equation(),
            ū[:, iplot, 1],
            nothing,
            t;
            reltol = 1e-4,
            abstol = 1e-6,
        ),
        ū[:, iplot, :],
        t,
    )
    function plot_loss(i, p, ifirst = 0)
        sol = solve_equation(filtered, ū[:, iplot, 1], p, t; reltol = 1e-4, abstol = 1e-6)
        err = relerr(sol, ū[:, iplot, :], t)
        println("Iteration $i \t average relative error $err")
        push!(hist_i, ifirst + i)
        push!(hist_err, err)
        pl = plot(; title = "Average relative error", xlabel = "Iterations")
        hline!(pl, [err_nomodel]; color = 1, linestyle = :dash, label = "No closure")
        plot!(pl, hist_i, hist_err; label = "With closure")
        display(pl)
    end
end

# Initial parameters
p = p₀
opt = Optimisers.setup(Optimisers.ADAM(0.001), p)
plot_loss(0, p)

# Derivative fitting loss
loss_df = create_loss_derivative_fit(
    # Merge solution and time dimension, with new size `(nx, nsolution*ntime)`
    reshape(dūdt_train, M, :),
    reshape(ū_train, M, :);

    # Number of random data samples for each loss evaluation (batch size)
    nuse = 100,

    # Tikhonov regularization weight
    λ = 1e-8,
);

# Fit predicted time derivatives to reference time derivatives
i_first = last(plot_loss.hist_i)
nplot = 100
for i ∈ 1:5000
    grad = first(gradient(loss_df, p))
    opt, p = Optimisers.update(opt, p, grad)
    # i % nplot == 0 ? plot_loss(i, p, i_first) : println("Iteration $i")
    i % nplot == 0 && plot_loss(i, p, i_first)
end
p_df = p

# filename = "output/$(eqname(equation()))_df.jld2"
# jldsave(filename; p_df)
# p_df = load(filename, "p_df")

# Trajectory fitting loss
loss_emb = create_loss_trajectory_fit(
    ū_train,
    t_train;

    # Number of random samples per evaluation
    nsample = 5,

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
);

# Fit the predicted trajectories to the reference trajectories
i_first = last(plot_loss.hist_i)
nplot = 100
for i ∈ 1:1000
    grad = first(gradient(loss_emb, p))
    opt, p = Optimisers.update(opt, p, grad)
    # i % nplot == 0 ? plot_loss(i, p, i_first) : println("Iteration $i")
    i % nplot == 0 && plot_loss(i, p, i_first)
end
p_tf = p

# filename = "output/$(eqname(equation()))_tf.jld2"
# jldsave(filename; p_tf)
# p_tf = load(filename, "p_tf")

## Plot performance evolution
# ū, u, t = ū_train, u_train, t_train
# ū, u, t = ū_valid, u_valid, t_valid
ū, u, t = ū_test, u_test, t_test
isample = 2:2
sol_nomodel =
    solve_equation(equation(), ū[:, isample, 1], nothing, t; reltol = 1e-4, abstol = 1e-6)
sol_df = solve_equation(filtered, ū[:, isample, 1], p_df, t; reltol = 1e-4, abstol = 1e-6)
sol_tf = solve_equation(filtered, ū[:, isample, 1], p_tf, t; reltol = 1e-4, abstol = 1e-6)
for (i, t) ∈ enumerate(t)
    pl = plot(;
        xlabel = "x",
        title = @sprintf("Solutions, t = %.3f", t),
        ylims = extrema(u[:, isample, :]),
    )

    # plot!(pl, ξ, u[:, isample, i]; linestyle = :dash, label = "Unfiltered")
    plot!(pl, x, ū[:, isample, i]; label = "Filtered exact")
    plot!(pl, x, sol_nomodel[i]; label = "No closure")
    # plot!(pl, x, sol_NN[i]; label = "Neural closure")
    plot!(pl, x, sol_df[i]; label = "Derivative fit")
    plot!(pl, x, sol_tf[i]; label = "Trajectory fit")

    display(pl)
    sleep(0.05)
end

# Momentum
m_exact = [Δx * sum(v) for v ∈ eachcol(ū[:, isample[1], :])]
m_nomodel = [Δx * sum(v) for v ∈ sol_nomodel.u]
m_df = [Δx * sum(v) for v ∈ sol_df.u]
m_tf = [Δx * sum(v) for v ∈ sol_tf.u]
pl = plot(; xlabel = "t", title = "Filtered Momentum")
plot!(t, m_exact; label = "Exact")
plot!(t, m_nomodel; label = "No closure")
plot!(t, m_df; label = "Derivative fit")
# plot!(t, m_tf; label = "Trajectory fit")
pl

# Energy
E_exact = [Δx * v'v / 2 for v ∈ eachcol(ū[:, isample[1], :])]
E_nomodel = [Δx * v'v / 2 for v ∈ eachcol(sol_nomodel[:, 1, :])]
E_df = [Δx * v'v / 2 for v ∈ eachcol(sol_df[:, 1, :])]
E_tf = [Δx * v'v / 2 for v ∈ eachcol(sol_tf[:, 1, :])]
pl = plot(; xlabel = "t", title = "Filtered Energy")
plot!(t, E_exact; label = "Exact")
plot!(t, E_nomodel; label = "No closure")
plot!(t, E_df; label = "Derivative fit")
plot!(t, E_tf; label = "Trajectory fit")
pl

plotsol(x, t, ū[:, isample[1], :])
plotsol(x, t, sol_df[:, 1, :])
plotsol(x, t, sol_tf[:, 1, :])
plotsol(x, t, sol_nomodel[:, 1, :])
plotsol(x, t, (sol_nomodel[:, 1, :] - ū[:, isample[1], :]) ./ norm(ū[:, isample[1], :]))
plotsol(x, t, (sol_df[:, 1, :] - ū[:, isample[1], :]) ./ norm(ū[:, isample[1], :]))
plotsol(x, t, (sol_tf[:, 1, :] - ū[:, isample[1], :]) ./ norm(ū[:, isample[1], :]))

relerr(sol_nomodel, ū[:, isample, :], t)
relerr(sol_df, ū[:, isample, :], t)
relerr(sol_tf, ū[:, isample, :], t)
