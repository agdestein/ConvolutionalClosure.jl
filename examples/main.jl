# # Closing the filtered Korteweg-De Vries equation
#
# In this example we learn a non-linear differential closure term for the
# filtered Korteweg-De Vries equation. The resulting model is a Universial
# differential equation (UDE).

using LinearAlgebra
using Lux
using Optimisers
using OrdinaryDiffEq
using Plots
using Printf
using Random
using SciMLSensitivity
using Zygote

using ConvolutionalClosure

# Domain length.
l() = 1.0

# Viscosity (if applicable)
μ() = 0.001

## Equation
# equation() = Convection(l())
# equation() = Burgers(l(), μ())
equation() = KortewegDeVries(l())
# equation() = KuramotoSivashinsky(l())

"""
    solve_equation(u₀, t; kwargs...)

Solve KdV equation from `t[1]` to `t[end]`.
The initial conditions `u₀` are of size `N × n_IC`.
The solution is saved at `t` of size `n_t`.

Return an `ODESolution` acting as an array of size `N × n_IC × n_t`.
"""
function solve_equation(u₀, t; kwargs...)
    problem = ODEProblem(equation(), u₀, extrema(t))
    solve(problem, Tsit5(); saveat = t, kwargs...)
end

# Spatial discretization
N = 200
x = LinRange(0, l(), N + 1)[2:end]
Δx = l() / N

# Discrete filter matrix
Δ = @. 8Δx * (1 + 1 / 3 * sin(2π * x))
W = sum(gaussian.(Δ, x .- x' .- z) for z ∈ -2:2)
W = W ./ sum(W; dims = 2)
heatmap(W; yflip = true, xmirror = true)

# Plot example solution
# u₀ = @. sinpi(2x) + sinpi(6x) + cospi(10x)
# u₀ = @. sin(2π * x)
u₀ = @. exp(-(x - 0.5)^2 / 0.005)
plot(u₀)
t = LinRange(0, 0.0002, 101)
sol = solve_equation(u₀, t; reltol = 1e-6, abstol = 1e-8)
Wsol = W * sol
for (i, t) ∈ enumerate(t)
    pl = plot(; xlabel = "x", title = @sprintf("t = %.5f", t), ylims = extrema(sol[:, :]))
    plot!(pl, x, sol[i]; label = "Unfiltered")
    plot!(pl, x, Wsol[:, i]; label = "Filtered")
    display(pl)
    sleep(0.05) # Time for plot pane to update
end

"""
Little hack: `Float64`-capable version of `Lux.glorot_uniform`.
This may break with future versions of Lux.
https://github.com/avik-pal/Lux.jl/blob/51bbf8dc489155c53f5f034b636848bdaabfc55d/src/utils.jl#L45-L48
"""
function glorot_uniform_T(rng::AbstractRNG, dims::Integer...; gain::Real = 1)
    scale = gain * sqrt(24.0 / sum(Lux._nfan(dims...)))
    return (rand(rng, dims...) .- 1 / 2) .* scale
end

## Safe (but creates NN with Float32)
# init_weight = Lux.glorot_uniform
# init_bias = Lux.zeros32

## This also works for Float64
init_weight = glorot_uniform_T
init_bias = zeros

# Discrete closure term for filtered equation
d₁ = 3
d₂ = 4
d₃ = 1
NN = Chain(
    # From (nx, nbatch) to (nx, nchannel, nbatch)
    u -> reshape(u, size(u, 1), 1, size(u, 2)),

    # Add channels
    u -> hcat(
        # Closure should depend on `u`
        u,

        # Square channel to mimic non-linear term
        u .* u,

        # Filter width channel for non-uniformity (same for each batch)
        repeat(Δ, 1, 1, size(u, 3)),
    ),

    # Manual padding to account for periodicity
    u -> [u[end-(d₁+d₂+d₃)+1:end, :, :]; u; u[1:(d₁+d₂+d₃), :, :]],

    # Some convolutional layers to mimic local differential operators
    Conv((2d₁ + 1,), 3 => 4, Lux.relu; init_weight),
    Conv((2d₂ + 1,), 4 => 3, Lux.relu; init_weight),
    Conv((2d₃ + 1,), 3 => 1; init_weight),

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
    du = equation()(u, p, t)
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
Closure for creating trajectory-fitting loss.
Chooses a random subset of the solutions and time points at each evaluation.
"""
function create_loss_embedded(
    u,
    t;
    n_sample = size(u, 2),
    n_time = length(t),
    λ = 0,
    kwargs...,
)
    function loss(p)
        iu = Zygote.@ignore sort(shuffle(1:size(u, 2))[1:n_sample])
        it = Zygote.@ignore sort(shuffle(1:length(t))[1:n_time])
        loss_embedded(solve_filtered, p, u[:, iu, it], t[it], λ; kwargs...)
    end
    loss
end

"""
Closure for creating derivative-fitting loss.
Chooses a random subset of the data samples each evaluation.
"""
function create_loss_derivative_fit(dudt, u; n_sample = size(u, 2), λ = 0)
    function loss(p)
        i = Zygote.@ignore sort(shuffle(1:size(u, 2))[1:n_sample])
        loss_derivative_fit(filtered, p, dudt[:, i], u[:, i], λ)
    end
    loss
end

# Fourier basis
K = 20
basis = [exp(2π * im * k * x) for x ∈ x, k ∈ -K:K]

# Fourier coefficients with random phase and amplitude
n_train = 100
n_test = 50
c_train = [randn() * exp(-2π * im * rand()) / (1 + abs(k)) for k ∈ -K:K, _ ∈ 1:n_train]
c_test = [randn() * exp(-2π * im * rand()) / (1 + abs(k)) for k ∈ -K:K, _ ∈ 1:n_test]

# Initial conditions (real-valued)
u₀_train = real.(basis * c_train)
u₀_test = real.(basis * c_test)

# Evaluation times
t_train = LinRange(0, 0.0001, 101)
t_test = LinRange(0, 0.0002, 61)

plot(x, u₀_train[:, 1:3]; xlabel = "x", title = "Initial conditions")

# Unfiltered solutions
u_train = solve_equation(u₀_train, t_train; reltol = 1e-4, abstol = 1e-6)
u_test = solve_equation(u₀_test, t_test; reltol = 1e-4, abstol = 1e-6)

# Filtered solutions
ū_train = apply_filter(W, u_train)
ū_test = apply_filter(W, u_test)

# Filtered time derivatives
dūdt_train = apply_filter(W, kdv(u_train, nothing, 0.0))
dūdt_test = apply_filter(W, kdv(u_test, nothing, 0.0))

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
    plot!(pl, x, u[:, iplot, i]; color = 1, label = "Unfiltered")
    plot!(pl, x, ū[:, iplot, i]; color = 2, label = "Filtered exact")
    display(pl)
    sleep(0.05)
end

# Callback for studying convergence
iplot = 1:10
hist_i = Int[]
hist_train = zeros(0)
hist_test = zeros(0)
sol_nomodel_train =
    solve_equation(ū_train[:, iplot, 1], t_train; reltol = 1e-4, abstol = 1e-6)
sol_nomodel_test =
    solve_equation(ū_test[:, iplot, 1], t_test; reltol = 1e-4, abstol = 1e-6)
err_nomodel_train = relerr(sol_nomodel_train, ū_train[:, iplot, :], t_train)
err_nomodel_test = relerr(sol_nomodel_test, ū_test[:, iplot, :], t_test)
function callback(i, p; i_first = 0)
    sol_train =
        solve_filtered(p, ū_train[:, iplot, 1], t_train; reltol = 1e-4, abstol = 1e-6)
    sol_test = solve_filtered(p, ū_test[:, iplot, 1], t_test; reltol = 1e-4, abstol = 1e-6)
    err_train = relerr(sol_train, ū_train[:, iplot, :], t_train)
    err_test = relerr(sol_test, ū_test[:, iplot, :], t_test)
    println("Iteration $i \t train $err_train \t test $err_test")
    push!(hist_i, i_first + i)
    push!(hist_train, err_train)
    push!(hist_test, err_test)
    pl = plot(; title = "Average relative error", xlabel = "Iterations")
    hline!(
        pl,
        [err_nomodel_train];
        color = 1,
        linestyle = :dash,
        label = "train, no closure",
    )
    plot!(pl, hist_i, hist_train; color = 1, label = "train, NN")
    hline!(pl, [err_nomodel_test]; color = 2, linestyle = :dash, label = "test, no closure")
    plot!(pl, hist_i, hist_test; color = 2, label = "test, NN")
    display(pl)
end

# Initial parameters
p = p₀
opt = Optimisers.setup(Optimisers.ADAM(0.001), p)
callback(0, p)

# Merge IC and time dimension, with new size N × (n_IC n_time). Use a subset of
# the data samples at each evaluation
loss_df = create_loss_derivative_fit(
    reshape(dūdt_train, N, :),
    reshape(ū_train, N, :);
    n_sample = 200,
    λ = 1e-6,
);

# Fit predicted time derivatives to reference time derivatives
i_first = last(hist_i)
n_cb = 50
for i ∈ 1:1000
    grad = first(gradient(loss_df, p))
    opt, p = Optimisers.update(opt, p, grad)
    i % n_cb == 0 ? callback(i, p; i_first) : println("Iteration $i")
end

# Fit the predicted trajectories to the reference trajectories using a subset
# of the data at each evaluation
loss_emb = create_loss_embedded(
    ū_train,
    t_train;
    n_sample = 10,
    n_time = 20,
    λ = 1.0e-6,
    # sensealg = BacksolveAdjoint(; autojacvec = ZygoteVJP()),
    sensealg = InterpolatingAdjoint(; autojacvec = ZygoteVJP()),
    # sensealg = QuadratureAdjoint(; autojacvec = ZygoteVJP()),
);

i_first = last(hist_i)
n_cb = 1
for i ∈ 1:10
    grad = first(gradient(loss_emb, p))
    opt, p = Optimisers.update(opt, p, grad)
    i % n_cb == 0 ? callback(i, p; i_first) : println("Iteration $i")
end

## Plot performance evolution
# ū, u, t = ū_train, u_train, t_train
ū, u, t = ū_test, u_test, t_test
sol = solve_equation(ū[:, :, 1], t; reltol = 1.0e-4, abstol = 1.0e-6)
sol_NN = solve_filtered(p, ū[:, :, 1], t; reltol = 1.0e-4, abstol = 1.0e-6)
isample = 2
for (i, t) ∈ enumerate(t)
    pl = plot(;
        xlabel = "x",
        title = @sprintf("Solutions, t = %.3f", t),
        ylims = extrema(u[:, isample, :]),
    )
    plot!(pl, x, u[:, isample, i]; linestyle = :dash, label = "Unfiltered")
    plot!(pl, x, ū[:, isample, i]; label = "Filtered exact")
    plot!(pl, x, sol[:, isample, i]; label = "No closure")
    plot!(pl, x, sol_NN[:, isample, i]; label = "Neural closure")
    display(pl)
    sleep(0.05)
end
