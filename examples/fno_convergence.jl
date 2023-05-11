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

"""
Domain length
"""
l() = 1.0

"""
Viscosity
"""
μ() = 0.001

"""
Reference simulation time
"""
tref() = 0.1

## Equation
# equation() = Convection(l())
# equation() = Diffusion(l(), μ())
equation() = Burgers(l(), μ())
# equation() = KortewegDeVries(l())
# equation() = KuramotoSivashinsky(l())
# equation() = Schrodinger()

discretize(N) = LinRange(0, l(), N + 1)[2:end]

"""
Filter width
"""
Δ() = 5 / 100 * l()

"""
Discrete filter matrix
"""
function filter_matrix(x, y)
    W = sum(-1:1) do z
        d = x .- y' .- z .* l()
        gaussian.(Δ(), d) .* (abs.(d) .≤ 3 ./ 2 .* Δ())
        # top_hat.(Δ, d)
    end
    W = W ./ sum(W; dims = 2)
    W = sparse(W)
    dropzeros!(W)
    W
end

# Fine discretization
N = 512
y = discretize(N)

# Training discretizations
MM = [32, 64, 128]
xx = discretize.(MM)

# Filter matrices
WW = [filter_matrix(x, y) for x ∈ xx]

plotmat(WW[1]; title = "W")

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

# ODE solver tolerances
tols = (; reltol = 1e-4, abstol = 1e-6)

# Unfiltered solutions
u_train = solve_equation(equation(), u₀_train, nothing, t_train; tols...)
u_valid = solve_equation(equation(), u₀_valid, nothing, t_valid; tols...)
u_test = solve_equation(equation(), u₀_test, nothing, t_test; tols...)

# Filtered solutions
v_train = [apply_matrix(W, u_train) for W in WW]
v_valid = [apply_matrix(W, u_valid) for W in WW]
v_test = [apply_matrix(W, u_test) for W in WW]

# Filtered time derivatives (for derivative fitting)
dvdt_train = [apply_matrix(W, equation()(u_train, nothing, 0.0)) for W in WW]
dvdt_valid = [apply_matrix(W, equation()(u_valid, nothing, 0.0)) for W in WW]
dvdt_test  = [apply_matrix(W, equation()(u_test, nothing, 0.0))  for W in WW]

# Callback for studying convergence
function create_callback(f, v, t; kwargs...)
    iplot = 1:10
    hist_i = Int[]
    hist_err = zeros(0)
    err_nomodel = relerr(
        Array(solve_equation(equation(), v[:, iplot, 1], nothing, t; kwargs...)),
        v[:, iplot, :],
    )
    function callback(i, p)
        sol = solve_equation(f, v[:, iplot, 1], p, t; kwargs...)
        err = relerr(Array(sol), v[:, iplot, :])
        println("Iteration $i \t average relative error $err")
        push!(hist_i, i)
        push!(hist_err, err)
        pl = plot(; title = "Average relative error", xlabel = "Iterations")
        hline!(pl, [err_nomodel]; color = 1, linestyle = :dash, label = "No closure")
        plot!(pl, hist_i, hist_err; label = "With closure")
        display(pl)
    end
end

# Initialize FNO
p₀_fno, c_fno = fourier_closure(
    # Latent dimension
    5,

    # Maximum frequency
    16;

    # Input channels
    channel_augmenter = u -> vcat(
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
filtered_fno(u, p, t) = equation()(u, nothing, t) + c_fno(u, p, t)

p_fno_df = train(
    # Loss function
    p -> prediction_loss(
        filtered_fno,
        p,

        # Merge solution and time dimension, with new size `(nx, nsolution*ntime)`
        reshape(dvdt_train, M, :),
        reshape(v_train, M, :);

        # Number of random data samples for each loss evaluation (batch size)
        nuse = 300,

        # Tikhonov regularization weight
        λ = 1e-8,
    ),

    # Initial parameters
    p₀_fno,
    # p_fno_df,

    # Iterations
    1_000;

    # Iterations per callback
    ncallback = 10,
    callback = create_callback(filtered_fno, v_valid, t_valid; tols...),
)

p_fno_tf = train(
    # Loss function
    p -> trajectory_loss(
        filtered_fno,
        p,
        v_train,
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

    # Initial parameters
    p_fno_df,

    # Iterations
    100;

    # Iterations per callback
    ncallback = 10,
    callback = create_callback(filtered_fno, v_valid, t_valid; tols...),
)

# p_fno = p_fno_df
p_fno = p_fno_tf

isample = 2
v, u, t = v_train[:, isample, :], u_train[:, isample, :], t_train
v, u, t = v_valid[:, isample, :], u_valid[:, isample, :], t_valid
v, u, t = v_test[:, isample, :], u_test[:, isample, :], t_test

relerr(Array(v_nomodel), v)
relerr(Array(v_simple), v)
relerr(Array(v_cnn), v)
relerr(Array(v_fno), v)

v, u, t = v_test, u_test, t_test
v_nomodel = solve_equation(equation(), v[:, :, 1], nothing, t; reltol = 1e-4, abstol = 1e-6)
v_simple =
    solve_equation(filtered_simple, v[:, :, 1], p_simple, t; reltol = 1e-4, abstol = 1e-6)
v_cnn = solve_equation(filtered_cnn, v[:, :, 1], p_cnn, t; reltol = 1e-4, abstol = 1e-6)
v_fno = solve_equation(filtered_fno, v[:, :, 1], p_fno, t; reltol = 1e-4, abstol = 1e-6)
