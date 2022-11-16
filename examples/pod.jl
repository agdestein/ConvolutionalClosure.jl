if isdefined(@__MODULE__, :LanguageServer)
    include("../src/ConvolutionalClosure.jl")
    using .ConvolutionalClosure
end

# # Closing a reduced order model (ROM)
#
# In this example we learn a non-linear differential closure term for a
# 1D equation projected onto an incomplete POD basis.

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
# l() = 8π
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

# Fine discretization
N = 200
ξ = LinRange(0, l(), N + 1)[2:end]
Δξ = l() / N

"""
Linear-ish frequency decay.
"""
decay(k) = 1 / (1 + abs(k))^1.2

# Maximum frequency in initial conditions
K = N ÷ 2
# K = 30

# Number of samples
n_train_pod = 120
n_train = 500
n_valid = 10
n_test = 60

# Initial conditions (real-valued)
u₀_train_pod = create_data(ξ, K, n_train; decay)
u₀_train = create_data(ξ, K, n_train; decay)
u₀_valid = create_data(ξ, K, n_valid; decay)
u₀_test = create_data(ξ, K, n_test; decay)
plot(ξ, u₀_train[:, 1:3]; xlabel = "x", title = "Initial conditions")

# Evaluation times
t_train_pod = LinRange(0, 10 * tref(), 60)
t_train = LinRange(0, tref(), 41)
t_valid = LinRange(0, 2 * tref(), 26)
t_test = LinRange(0, 10 * tref(), 61)

# FOM solutions
u_train_pod = solve_equation(equation(), u₀_train_pod, nothing, t_train_pod; reltol = 1e-4, abstol = 1e-6)
u_train     = solve_equation(equation(), u₀_train,     nothing, t_train; reltol = 1e-4, abstol = 1e-6)
u_valid     = solve_equation(equation(), u₀_valid,     nothing, t_valid; reltol = 1e-4, abstol = 1e-6)
u_test      = solve_equation(equation(), u₀_test,      nothing, t_test; reltol = 1e-4, abstol = 1e-6)

# Number of POD modes
M = 30

# POD
rom = create_pod(equation(), u_train_pod, M)
(; Φ) = rom
Φ

# ROM solutions
v_train = apply_filter(Φ', u_train)
v_valid = apply_filter(Φ', u_valid)
v_test = apply_filter(Φ', u_test)

# ROM solutions
dvdt_train = apply_filter(Φ', equation()(u_train, nothing, 0.0))
dvdt_valid = apply_filter(Φ', equation()(u_valid, nothing, 0.0))
dvdt_test = apply_filter(Φ', equation()(u_test, nothing, 0.0))

## Example solution
u₀ = @. sin(2π * ξ / l()) + sin(2π * 3ξ / l()) + cos(2π * 5ξ / l())
u₀ = @. sin(2π * ξ / l())
u₀ = @. exp(-(ξ / l() - 0.5)^2 / 0.005)
plot(u₀; xlabel = "x")

# Plot example solution
t = LinRange(0, 50 * tref(), 101)
sol_fom = solve_equation(equation(), u₀, nothing, t; reltol = 1e-6, abstol = 1e-8)
sol_rom = solve_equation(rom, Φ' * u₀, nothing, t; reltol = 1e-6, abstol = 1e-8)
for (i, t) ∈ enumerate(t)
    pl = plot(; xlabel = "x", title = @sprintf("t = %.2f", t), ylims = extrema(sol_fom[:, :]))
    plot!(pl, ξ, sol_fom[i]; label = "FOM")
    plot!(pl, ξ, Φ * sol_rom[i]; label = "ROM")
    display(pl)
    sleep(0.05) # Time for plot pane to update
end

# Momentum
m_fom = [Δξ * sum(u) for u ∈ sol_fom.u]
m_rom = [Δξ * sum(Φ * v) for v ∈ sol_rom.u]
plot(t, [m_fom, m_rom]; xlabel = "t", title = "Momentum", label = ["FOM" "ROM"])

# Energy
E_fom = [Δξ * u'u / 2 for u ∈ sol_fom.u]
E_rom = [Δξ * v'v / 2 for v ∈ sol_rom.u]
plot(t, [E_fom E_rom]; xlabel = "t", title = "Energy", label = ["FOM" "ROM"])

## Safe (but creates NN with Float32)
# init_weight = Lux.glorot_uniform
# init_bias = Lux.zeros32

# This also works for Float64
init_weight = glorot_uniform_Float64
init_bias = zeros

# Layer sizes (nlayer)
r = [2M, M, M, M]

# Activation functions (nlayer)
a = [Lux.relu, Lux.relu, identity]

# Discrete closure term for filtered equation
NN = Chain(
    # Create input channels
    u -> vcat(
        # Vanilla channel
        u,

        # Square channel to mimic non-linear term
        u .* u,
    ),

    # Some dense layers
    (Dense(r[i], r[i+1], a[i]; init_weight) for i ∈ eachindex(a))...,
)

# Initialize NN
rng = Random.default_rng()
Random.seed!(rng, 0)
params, state = Lux.setup(rng, NN)
p₀, re = Lux.destructure(params)

"""
Compute closure term for given parameters `p`.
"""
closure(v, p, t) = first(Lux.apply(NN, v, re(p), state))

"""
Compute right hand side of closed filtered equation.
This is modeled as unfiltered RHS + neural closure term.
"""
function rom_closed(v, p, t)
    dv = rom(v, nothing, t)
    c = closure(v, p, t)
    dv + c
    # c
end

"""
Creating derivative-fitting loss function.
Chooses a random subset (`nuse`) of the data samples at each evaluation.
Note that both `u` and `dudt` are of size `(nx, nsample)`.
"""
function create_loss_derivative_fit(dvdt, v; nuse = size(v, 2), λ = 0)
    function loss(p)
        i = Zygote.@ignore sort(shuffle(1:size(v, 2))[1:nuse])
        loss_derivative_fit(rom_closed, p, dvdt[:, i], v[:, i], λ)
    end
    loss
end

"""
Create trajectory-fitting loss function.
Chooses a random subset of the solutions and time points at each evaluation.
Note that `u` is of size `(nx, nsolution, ntime)`.
"""
function create_loss_trajectory_fit(
    v,
    t;
    nsolution = size(v, 2),
    ntime = length(t),
    λ = 0,
    kwargs...,
)
    function loss(p)
        iv = Zygote.@ignore sort(shuffle(1:size(v, 2))[1:nsolution])
        it = Zygote.@ignore sort(shuffle(1:length(t))[1:ntime])
        loss_embedded(rom_closed, p, v[:, iv, it], t[it], λ; kwargs...)
    end
    loss
end

## Plot some reference solutions
u, v, t = u_train, v_train, t_train;
# u, v, t = u_valid, v_valid, t_valid;
# u, v, t = u_test, v_test, t_test;
iplot = 1:3
for (i, t) ∈ enumerate(t)
    pl = plot(;
        xlabel = "x",
        title = @sprintf("Reference data, t = %.5f", t),
        ylims = extrema(u[:, iplot, :]),
    )
    plot!(pl, ξ, u[:, iplot, i]; color = 1, label = "FOM")
    plot!(pl, ξ, Φ * v[:, iplot, i]; color = 2, label = "ROM")
    display(pl)
    sleep(0.05)
end

# Callback for studying convergence
plot_loss = let
    v, t = v_valid, t_valid
    # v, t = v_test, t_test
    iplot = 1:10
    hist_i = Int[]
    hist_err = zeros(0)
    err_nomodel = relerr(
        solve_equation(rom, v[:, iplot, 1], nothing, t; reltol = 1e-4, abstol = 1e-6),
        v[:, iplot, :],
        t,
    )
    function plot_loss(i, p, ifirst = 0)
        sol = solve_equation(rom_closed, v[:, iplot, 1], p, t; reltol = 1e-4, abstol = 1e-6)
        err = relerr(sol, v[:, iplot, :], t)
        println("Iteration $i \t average relative error $err")
        push!(hist_i, ifirst + i)
        push!(hist_err, err)
        pl = plot(;
            title = "Average relative error",
            xlabel = "Iterations",
        )
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
    # Merge sample and time dimension, with new size `(nx, nsample*ntime)`
    reshape(dvdt_train, M, :),
    reshape(v_train, M, :);

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
    v_train,
    t_train;

    # Number of random samples per evaluation
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
);

# Fit the predicted trajectories to the reference trajectories
i_first = last(plot_loss.hist_i)
nplot = 1
for i ∈ 1:100
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
# u, v, t = u_train, v_train, t_train
# u, v, t = u_valid, v_valid, t_valid
u, v, t = u_test, v_test, t_test
isample = 2:2
sol_rom = solve_equation(rom, v[:, isample, 1], nothing, t; reltol = 1e-4, abstol = 1e-6)
sol_df = solve_equation(rom_closed, v[:, isample, 1], p_df, t; reltol = 1e-4, abstol = 1e-6)
sol_tf = solve_equation(rom_closed, v[:, isample, 1], p_tf, t; reltol = 1e-4, abstol = 1e-6)
for (i, t) ∈ enumerate(t)
    pl = plot(;
        xlabel = "x",
        title = @sprintf("Solutions, t = %.3f", t),
        ylims = extrema(u[:, isample, :]),
    )

    # plot!(pl, ξ, u[:, isample, i]; linestyle = :dash, label = "Unfiltered")
    plot!(pl, ξ, Φ * v[:, isample, i]; label = "Exact ROM")
    plot!(pl, ξ, Φ * sol_rom[i]; label = "No closure")
    plot!(pl, ξ, Φ * sol_df[i]; label = "Derivative fit")
    # plot!(pl, x, Φ * sol_tf[i]; label = "Trajectory fit")

    display(pl)
    sleep(0.05)
end

# Momentum
m_fom = [Δξ * sum(Φ * v) for v ∈ eachcol(v[:, isample[1], :])]
m_rom = [Δξ * sum(Φ * v) for v ∈ sol_rom.u]
m_df = [Δξ * sum(Φ * v) for v ∈ sol_df.u]
m_tf = [Δξ * sum(Φ * v) for v ∈ sol_tf.u]
pl = plot(; xlabel = "t", title = "Momentum")
plot!(t, m_fom; label = "FOM")
plot!(t, m_rom; label = "ROM")
plot!(t, m_df; label = "Derivative fit")
# plot!(t, m_tf; label = "Trajectory fit")
pl

# Energy
E_fom = [Δξ * v'v / 2 for v ∈ eachcol(v[:, isample[1], :])]
E_rom = [Δξ * v'v / 2 for v ∈ eachcol(sol_rom[:, 1, :])]
E_df = [Δξ * v'v / 2 for v ∈ eachcol(sol_df[:, 1, :])]
E_tf = [Δξ * v'v / 2 for v ∈ eachcol(sol_tf[:, 1, :])]
pl = plot(; xlabel = "t", title = "Energy")
plot!(t, E_fom; label = "FOM")
plot!(t, E_rom; label = "ROM")
plot!(t, E_df; label = "Derivative fit")
# plot!(t, E_tf; label = "Trajectory fit")
pl

plotsol(ξ, t, u[:, isample[1], :])
plotsol(ξ, t, Φ * v[:, isample[1], :])
plotsol(ξ, t, Φ * sol_rom[:, 1, :])
plotsol(ξ, t, Φ * sol_df[:, 1, :])
plotsol(ξ, t, Φ * sol_tf[:, 1, :])
plotsol(ξ, t, Φ * (sol_rom[:, 1, :] - v[:, isample[1], :]) ./ norm(v[:, isample[1], :]))
plotsol(ξ, t, Φ * (sol_df[:, 1, :] - v[:, isample[1], :]) ./ norm(v[:, isample[1], :]))
plotsol(ξ, t, Φ * (sol_tf[:, 1, :] - v[:, isample[1], :]) ./ norm(v[:, isample[1], :]))

plotsol(1:M, t, v[:, isample[1], :])
plotsol(1:M, t, sol_rom[:, 1, :])
plotsol(1:M, t, sol_df[:, 1, :])
plotsol(1:M, t, sol_tf[:, 1, :])
plotsol(1:M, t, (sol_rom[:, 1, :] - v[:, isample[1], :]) ./ norm(v[:, isample[1], :]))
plotsol(1:M, t, (sol_df[:, 1, :] - v[:, isample[1], :]) ./ norm(v[:, isample[1], :]))
plotsol(1:M, t, (sol_tf[:, 1, :] - v[:, isample[1], :]) ./ norm(v[:, isample[1], :]))

relerr(sol_rom, v[:, isample, :], t)
relerr(sol_df, v[:, isample, :], t)
relerr(sol_tf, v[:, isample, :], t)
