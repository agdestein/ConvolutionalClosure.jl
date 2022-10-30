# # Closing a filtered equation
#
# In this example we learn a non-linear differential closure term for a
# filtered 1D equation.

using ConvolutionalClosure
using LinearAlgebra
using Lux
using Optimisers
using OrdinaryDiffEq
using Plots
using Printf
using Random
using SciMLSensitivity
using Zygote

# Domain length.
l() = 8π

# Viscosity (if applicable)
μ() = 0.001

# Reference simulation time
tref() = 0.1

## Equation
# equation() = Convection(l())
# equation() = Burgers(l(), μ())
equation() = KortewegDeVries(l())
# equation() = KuramotoSivashinsky(l())

"""
    solve_equation(u₀, t; kwargs...)

Solve equation from `t[1]` to `t[end]`.
The initial conditions `u₀` are of size `N × n_IC`.
The solution is saved at `t` of size `n_t`.

Return an `ODESolution` acting as an array of size `N × n_IC × n_t`.
"""
function solve_equation(u₀, t; kwargs...)
    problem = ODEProblem(equation(), u₀, extrema(t))
    solve(problem, Tsit5(); saveat = t, kwargs...)
end

# Fine discretization
N = 200
ξ = LinRange(0, l(), N + 1)[2:end]
Δξ = l() / N

# Coarse discretization
M = 100
x = LinRange(0, l(), M + 1)[2:end]
Δx = l() / N

# # Filter widths
# Δ = @. 4Δx * (1 + 1 / 3 * sin(2π * x / l()))
# plot(x, Δ; xlabel = "x", title = "Filter width")

# Filter width
Δ = 3Δx

# Discrete filter matrix
W = sum(gaussian.(Δ, x .- ξ' .- z .* l()) for z ∈ -2:2)
W = W ./ sum(W; dims = 2)
heatmap(W; yflip = true, xmirror = true, title = "Discrete filter")

## Example solution
u₀ = @. sin(2π * ξ / l()) + sin(2π * 3ξ / l()) + cos(2π * 5ξ / l())
u₀ = @. sin(2π * ξ / l())
u₀ = @. exp(-(ξ / l() - 0.5)^2 / 0.005)
plot(u₀; xlabel = "x")

# Plot example solution
t = LinRange(0, 10 * tref(), 101)
sol = solve_equation(u₀, t; reltol = 1e-6, abstol = 1e-8)
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

"""
Generate random weights as in `Lux.glorot_uniform`, but with Float64.
https://github.com/avik-pal/Lux.jl/blob/51bbf8dc489155c53f5f034b636848bdaabfc55d/src/utils.jl#L45-L48
"""
function glorot_uniform_Float64(rng::AbstractRNG, dims::Integer...; gain::Real = 1)
    scale = gain * sqrt(24 / sum(Lux._nfan(dims...)))
    return (rand(rng, dims...) .- 1 / 2) .* scale
end

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

    # From (nx, nchannel = 1, nbatch) to (nx, nbatch)
    u -> reshape(u, size(u, 1), size(u, 3)),
)
NN

# Get parameter structure for NN
rng = Random.default_rng()
Random.seed!(rng, 0)
params, state = Lux.setup(rng, NN)
p₀, re = Lux.destructure(params)

"""
Compute right hand side of closed filtered equation.
This is modeled as unfiltered RHS + neural closure term.
"""
function filtered(u, p, t)
    du = equation()(u, nothing, t)
    closure = first(Lux.apply(NN, u, re(p), state))
    du + closure
end

"""
Solve filtered equation starting from filtered initial conditions.
"""
function solve_filtered(p, u₀, t; kwargs...)
    problem = ODEProblem(filtered, u₀, extrema(t), p)
    solve(problem, Tsit5(); saveat = t, kwargs...)
end

"""
Closure for creating derivative-fitting loss.
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
Closure for creating trajectory-fitting loss.
Chooses a random subset of the solutions and time points at each evaluation.
Note that `u` is of size `(nx, nsample, ntime)`
"""
function create_loss_embedded(
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
        loss_embedded(solve_filtered, p, u[:, iu, it], t[it], λ; kwargs...)
    end
    loss
end

# Maximum frequency in initial conditions
K = 30

# Fourier basis
basis = [exp(2π * im * k * ξ / l()) for ξ ∈ ξ, k ∈ -K:K]

# Number of training and testing samples
n_train = 120
n_test = 60

"""
Linear-ish frequency decay.
"""
decay(k) = 1 / (1 + abs(k))

# Fourier coefficients with random phase and amplitude
c_train = [randn() * exp(-2π * im * rand()) * decay(k) for k ∈ -K:K, _ ∈ 1:n_train]
c_test = [randn() * exp(-2π * im * rand()) * decay(k) for k ∈ -K:K, _ ∈ 1:n_test]

# Initial conditions (real-valued)
u₀_train = real.(basis * c_train)
u₀_test = real.(basis * c_test)
plot(ξ, u₀_train[:, 1:3]; xlabel = "x", title = "Initial conditions")

# Evaluation times
t_train = LinRange(0, tref(), 41)
t_test = LinRange(0, 2 * tref(), 61)

# Unfiltered solutions
u_train = solve_equation(u₀_train, t_train; reltol = 1e-4, abstol = 1e-6)
u_test = solve_equation(u₀_test, t_test; reltol = 1e-4, abstol = 1e-6)

# Filtered solutions
ū_train = apply_filter(W, u_train)
ū_test = apply_filter(W, u_test)

# Filtered time derivatives (for derivative fitting)
dūdt_train = apply_filter(W, equation()(u_train, nothing, 0.0))
dūdt_test = apply_filter(W, equation()(u_test, nothing, 0.0))

## Plot some reference solutions
u, ū, t = u_train, ū_train, t_train
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
    iplot = 1:10
    hist_i = Int[]
    hist_err = zeros(0)
    err_nomodel = relerr(
        solve_equation(ū_test[:, iplot, 1], t_test; reltol = 1e-4, abstol = 1e-6),
        ū_test[:, iplot, :],
        t_test,
    )
    function (i, p, ifirst = 0)
        sol = solve_filtered(p, ū_test[:, iplot, 1], t_test; reltol = 1e-4, abstol = 1e-6)
        err = relerr(sol, ū_test[:, iplot, :], t_test)
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
    # Merge sample and time dimension, with new size nx × (nsample ntime)
    reshape(dūdt_train, M, :),
    reshape(ū_train, M, :);

    # Number of random data samples for each loss evaluation (batch size)
    nuse = 100,

    # Tikhonov regularization weight
    λ = 1e-8,
);

# Fit predicted time derivatives to reference time derivatives
i_first = last(plot_loss.hist_i)
nplot = 50
for i ∈ 1:5000
    grad = first(gradient(loss_df, p))
    opt, p = Optimisers.update(opt, p, grad)
    i % nplot == 0 ? plot_loss(i, p, i_first) : println("Iteration $i")
end

# Trajectory fitting loss
loss_emb = create_loss_embedded(
    ū_train,
    t_train;

    # Number of random samples per evaluation
    nsample = 10,

    # Number of random time instances per evaluation
    ntime = 20,

    # Tikhonov regularization weight
    λ = 1.0e-8,

    ## Sensitivity algorithm for computing gradient
    # sensealg = BacksolveAdjoint(; autojacvec = ZygoteVJP()),
    sensealg = InterpolatingAdjoint(; autojacvec = ZygoteVJP()),
    # sensealg = QuadratureAdjoint(; autojacvec = ZygoteVJP()),
);

# Fit the predicted trajectories to the reference trajectories
i_first = last(plot_loss.hist_i)
n_cb = 1
for i ∈ 1:100
    grad = first(gradient(loss_emb, p))
    opt, p = Optimisers.update(opt, p, grad)
    i % n_cb == 0 ? plot_loss(i, p, i_first) : println("Iteration $i")
end

## Plot performance evolution
# ū, u, t = ū_train, u_train, t_train
ū, u, t = ū_test, u_test, t_test
sol = solve_equation(ū[:, :, 1], t; reltol = 1e-4, abstol = 1e-6)
sol_NN = solve_filtered(p, ū[:, :, 1], t; reltol = 1e-4, abstol = 1e-6)
isample = 2
for (i, t) ∈ enumerate(t)
    pl = plot(;
        xlabel = "x",
        title = @sprintf("Solutions, t = %.3f", t),
        ylims = extrema(u[:, isample, :]),
    )
    plot!(pl, ξ, u[:, isample, i]; linestyle = :dash, label = "Unfiltered")
    plot!(pl, x, ū[:, isample, i]; label = "Filtered exact")
    plot!(pl, x, sol[:, isample, i]; label = "No closure")
    plot!(pl, x, sol_NN[:, isample, i]; label = "Neural closure")
    display(pl)
    sleep(0.05)
end
