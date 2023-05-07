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

"Domain length"
l() = 1.0

"Viscosity"
μ() = 0.001

"Reference simulation time"
tref() = 0.1

## Equation
# equation() = Convection(l())
# equation() = Diffusion(l(), μ())
equation() = Burgers(l(), μ())
# equation() = KortewegDeVries(l())
# equation() = KuramotoSivashinsky(l())
# equation() = Schrodinger()

# Coarse discretization
M = 32

# Fine discretization
s = 8
N = s * M

# Grid
x = LinRange(0, l(), M + 1)[2:end]
y = LinRange(0, l(), N + 1)[2:end]
Δx = l() / M
Δy = l() / N

# Filter widths
ΔΔ(x) = 5 / 100 * l() * (1 + 1 / 3 * sin(2π * x / l()))
Δ = ΔΔ.(x)

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

Φ, σ, Ψ = svd(Matrix(W))
Σ = Diagonal(σ)

i = 7
plot(x, Φ[:, i])
plot!(y, √s * Ψ[:, i])

"""
Linear-ish frequency decay.
"""
decay(k) = 1 / (1 + abs(k))^1.2

## Maximum frequency in initial conditions
# K = N ÷ 2
K = 30

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
v_train = apply_matrix(W, u_train)
v_valid = apply_matrix(W, u_valid)
v_test = apply_matrix(W, u_test)

# Filtered time derivatives (for derivative fitting)
dvdt_train = apply_matrix(W, equation()(u_train, nothing, 0.0))
dvdt_valid = apply_matrix(W, equation()(u_valid, nothing, 0.0))
dvdt_test = apply_matrix(W, equation()(u_test, nothing, 0.0))

# Callback for studying convergence
function create_callback(f, v, t)
    iplot = 1:10
    hist_i = Int[]
    hist_err = zeros(0)
    err_nomodel = relerr(
        Array(
            solve_equation(
                equation(),
                v[:, iplot, 1],
                nothing,
                t;
                reltol = 1e-4,
                abstol = 1e-6,
            ),
        ),
        v[:, iplot, :],
    )
    function callback(i, p)
        sol = solve_equation(f, v[:, iplot, 1], p, t; reltol = 1e-4, abstol = 1e-6)
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

# Reference model
c_ref(u) = W * equation()(u, nothing, 0.0) - equation()(W * u, nothing, 0.0)

# Initialize NN
p₀_cnn, c_cnn = convolutional_closure(
    # Kernel radii
    [5, 5, 3],

    # Number of channels
    # Last must be 1
    [4, 3, 1],

    # Activation functions
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

# FNO
p₀_fno, c_fno = fourier_closure(
    # Latent dimension
    5,

    # Maximum frequency
    12;

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

# MTO
p₀_mto, c_mto = matrix_transform_closure(
    # Matrix transform 
    Φ[:, 1:10],

    # Latent dimension
    5;

    # Input channels
    channel_augmenter = u -> vcat(
        # Vanilla channel
        u,

        # Square channel to mimic non-linear term
        u .* u,
    ),
)

"""
Compute right hand side of closed filtered equation.
This is modeled as unfiltered RHS + neural closure term.
"""
filtered_simple(u, p, t) = equation()(u, nothing, t) + c_simple(u, p, t)
filtered_cnn(u, p, t) = equation()(u, nothing, t) + c_cnn(u, p, t)
filtered_fno(u, p, t) = equation()(u, nothing, t) + c_fno(u, p, t)
filtered_mto(u, p, t) = equation()(u, nothing, t) + c_mto(u, p, t)

p_simple_df = train(
    # Loss function
    p -> derivative_loss(
        filtered_simple,
        p,

        # Merge solution and time dimension, with new size `(nx, nsolution*ntime)`
        reshape(dvdt_train, M, :),
        reshape(v_train, M, :);

        # Number of random data samples for each loss evaluation (batch size)
        nuse = 100,

        # Tikhonov regularization weight
        λ = 1e-8,
    ),

    # Initial parameters
    p₀_simple,
    # p_simple_df,

    # Iterations
    10_000;

    # Iterations per callback
    ncallback = 100,
    callback = create_callback(filtered_simple, v_valid, t_valid),
)

p_cnn_df = train(
    # Loss function
    p -> derivative_loss(
        filtered_cnn,
        p,

        # Merge solution and time dimension, with new size `(nx, nsolution*ntime)`
        reshape(dvdt_train, M, :),
        reshape(v_train, M, :);

        # Number of random data samples for each loss evaluation (batch size)
        nuse = 100,

        # Tikhonov regularization weight
        λ = 1e-8,
    ),

    # Initial parameters
    p₀_cnn,
    
    # Iterations
    10_000;

    # Iterations per callback
    ncallback = 100,
    callback = create_callback(filtered_cnn, v_valid, t_valid),
)

p_fno_df = train(
    # Loss function
    p -> derivative_loss(
        filtered_fno,
        p,

        # Merge solution and time dimension, with new size `(nx, nsolution*ntime)`
        reshape(dvdt_train, M, :),
        reshape(v_train, M, :);

        # Number of random data samples for each loss evaluation (batch size)
        nuse = 100,

        # Tikhonov regularization weight
        λ = 1e-8,
    ),

    # Initial parameters
    p₀_fno,
    # p_fno_df,

    # Iterations
    5_000;

    # Iterations per callback
    ncallback = 10,
    callback = create_callback(filtered_fno, v_valid, t_valid),
)

p_mto_df = train(
    # Loss function
    p -> derivative_loss(
        filtered_mto,
        p,

        # Merge solution and time dimension, with new size `(nx, nsolution*ntime)`
        reshape(dvdt_train, M, :),
        reshape(v_train, M, :);

        # Number of random data samples for each loss evaluation (batch size)
        nuse = 100,

        # Tikhonov regularization weight
        λ = 1e-8,
    ),

    # Initial parameters
    p₀_mto,
    # p_mto_df,

    # Iterations
    1_000;

    # Iterations per callback
    ncallback = 10,
    callback = create_callback(filtered_mto, v_valid, t_valid),
)

uu = u₀_test[:, 1]
vv = W * uu

plot(; xlabel = "x", title = "Closure term")
# plot!(x, equation()(vv, nothing, 0.0); label = "vv")
plot!(x, c_ref(uu); label = "Ref")
plot!(x, c_cnn(vv, p_cnn_df, 0.0); label = "CNN")
plot!(x, c_fno(vv, p_fno_df, 0.0); label = "FNO")
plot!(x, c_mto(vv, p_mto_df, 0.0); label = "MTO")

# filename = "output/$(eqname(equation()))_df.jld2"
# jldsave(filename; p_df)
# p_df = load(filename, "p_df")

p_simple_tf = train(
    # Loss function
    p -> trajectory_loss(
        filtered_simple,
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

    # Initial paramters
    p_simple_df,

    # Iterations
    100;

    # Iterations per callback
    ncallback = 10,
    callback = create_callback(filtered_simple, v_valid, t_valid),
)

p_cnn_tf = train(
    # Loss function
    p -> trajectory_loss(
        filtered_cnn,
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
    p_cnn_df,

    # Iterations
    100;

    # Iterations per callback
    ncallback = 10,
    callback = create_callback(filtered_cnn, v_valid, t_valid),
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
    callback = create_callback(filtered_fno, v_valid, t_valid),
)

# p_simple = p_simple_df
# p_cnn    = p_cnn_df
# p_fno    = p_fno_df
p_simple = p_simple_tf
p_cnn = p_cnn_tf
p_fno = p_fno_tf

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
v = W * u

isample = 2
v, u, t = v_train[:, isample, :], u_train[:, isample, :], t_train
v, u, t = v_valid[:, isample, :], u_valid[:, isample, :], t_valid
v, u, t = v_test[:, isample, :], u_test[:, isample, :], t_test

v_nomodel = solve_equation(equation(), v[:, 1], nothing, t; reltol = 1e-4, abstol = 1e-6)
v_simple =
    solve_equation(filtered_simple, v[:, 1], p_simple, t; reltol = 1e-4, abstol = 1e-6)
v_cnn = solve_equation(filtered_cnn, v[:, 1], p_cnn, t; reltol = 1e-4, abstol = 1e-6)
v_fno = solve_equation(filtered_fno, v[:, 1], p_fno, t; reltol = 1e-4, abstol = 1e-6)
for (i, t) ∈ enumerate(t)
    pl =
        plot(; xlabel = "x", title = @sprintf("Solutions, t = %.3f", t), ylims = extrema(v))
    # plot!(pl, y, u[:, i]; linestyle = :dash, label = "Unfiltered")
    plot!(pl, x, v[:, i]; label = "Filtered exact")
    plot!(pl, x, v_nomodel[i]; label = "No closure")
    plot!(pl, x, v_simple[i]; label = "Linear")
    plot!(pl, x, v_cnn[i]; label = "CNN")
    plot!(pl, x, v_fno[i]; label = "FNO")
    # plot!(pl, x, sol_tf[i]; label = "Trajectory fit")
    display(pl)
    sleep(0.05)
end

# Momentum
m_exact = [Δx * sum(v) for v ∈ eachcol(v)]
m_nomodel = [Δx * sum(v) for v ∈ v_nomodel.u]
m_simple = [Δx * sum(v) for v ∈ v_simple.u]
m_cnn = [Δx * sum(v) for v ∈ sol.u]
m_fno = [Δx * sum(v) for v ∈ sol_fno.u]
# m_tf = [Δx * sum(v) for v ∈ sol_tf.u]
pl = plot(; xlabel = "t", title = "Filtered Momentum")
plot!(t, m_exact; label = "Exact")
plot!(t, m_nomodel; label = "No closure")
plot!(t, m_simple; label = "Linear")
plot!(t, m_cnn; label = "CNN")
plot!(t, m_fno; label = "FNO")
# plot!(t, m_tf; label = "Trajectory fit")
pl

# Energy
E_exact = [Δx * v'v / 2 for v ∈ eachcol(v)]
E_nomodel = [Δx * v'v / 2 for v ∈ v_nomodel.u]
E_simple = [Δx * v'v / 2 for v ∈ v_simple.u]
E_cnn = [Δx * v'v / 2 for v ∈ v_cnn.u]
E_fno = [Δx * v'v / 2 for v ∈ v_fno.u]
# E_tf = [Δx * v'v / 2 for v ∈ sol_tf.u]
pl = plot(; xlabel = "t", title = "Filtered Energy")
plot!(t, E_exact; label = "Exact")
plot!(t, E_nomodel; label = "No closure")
plot!(t, E_simple; label = "Linear")
plot!(t, E_cnn; label = "CNN")
plot!(t, E_fno; label = "FNO")
# plot!(t, E_tf; label = "Trajectory fit")
pl

plotsol(y, t, u; title = "u")
# savefig(loc * "$(eqname(equation()))_u.png")
plotsol(x, t, v; title = "Wu")
# savefig(loc * "$(eqname(equation()))_Wu.png")
plotsol(x, t, v_cnn; title = "CNN")
plotsol(x, t, v_fno; title = "FNO")
plotsol(x, t, v_nomodel)
plotsol(x, t, (v_nomodel - v) ./ norm(v))
plotsol(x, t, (v_simple - v) ./ norm(v))
plotsol(x, t, (v_cnn - v) ./ norm(v))
plotsol(x, t, (v_fno - v) ./ norm(v))

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
plot!(t, errerr(v_nomodel, v); label = "No model")
plot!(t, errerr(v_simple, v); label = "Linear")
plot!(t, errerr(v_cnn, v); label = "CNN")
plot!(t, errerr(v_fno, v); label = "FNO")
pl

# savefig(loc * "burgers_relative_errors.pdf")
