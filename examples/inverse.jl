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
t() = 0.1

## Equation
# equation() = Convection(l())
# equation() = Diffusion(l(), μ())
equation() = Burgers(l(), μ())
# equation() = KortewegDeVries(l())
# equation() = KuramotoSivashinsky(l())
# equation() = Schrodinger()

# Coarse discretization
M = 50

# Fine discretization
s = 10
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

"""
Linear-ish frequency decay.
"""
decay(k) = 1 / (1 + abs(k))^1.2

## Maximum frequency in initial conditions
# K = N ÷ 2
K = 10

# Number of samples
n_train = 1000
n_valid = 100
n_test = 300

# Initial conditions (real-valued)
u₀_train = create_data(y, K, n_train; decay)
u₀_valid = create_data(y, K, n_valid; decay)
u₀_test = create_data(y, K, n_test; decay)
plot(y, u₀_train[:, 1:3]; xlabel = "x", title = "Initial conditions")

# Unfiltered solutions
u_train = solve_equation(equation(), u₀_train, nothing, (0, t()); reltol = 1e-4, abstol = 1e-6)
u_valid = solve_equation(equation(), u₀_valid, nothing, (0, t()); reltol = 1e-4, abstol = 1e-6)
u_test = solve_equation(equation(), u₀_test, nothing, (0, t()); reltol = 1e-4, abstol = 1e-6)

# Filtered solutions
v_train = apply_matrix(W, u_train)
v_valid = apply_matrix(W, u_valid)
v_test = apply_matrix(W, u_test)

# Callback for studying convergence
function create_callback(f, xy, t)
    iplot = 1:10
    hist_i = Int[]
    hist_err = zeros(0)
    x, y = xy[:, iplot, 2], xy[:, iplot, 1]
    function callback(i, p)
        z = f(y, p, t)
        err = relerr(z, y)
        println("Iteration $i \t average relative error $err")
        push!(hist_i, i)
        push!(hist_err, err)
        pl = plot(; title = "Average relative error", xlabel = "Iterations")
        plot!(pl, hist_i, hist_err; label = "FNO")
        display(pl)
    end
end

# FNO
p₀_fno, fno = fourier_closure(
    # Latent dimension
    30,

    # Maximum frequency
    20;

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

fno(u_train[:, 1:3, 2], p₀_fno, t())

ff(p) = sum(fno(u_train[:, 1:10, 2], p, t()))

@time gradient(ff, p₀_fno)

length(p₀_fno)

p_fno = train(
    # Loss function
    p -> prediction_loss(
        fno,
        p,

        # Input data (final conditions)
        u_train[:, :, 2],

        # Output data (initial conditions)
        u_train[:, :, 1];

        # Number of random data samples for each loss evaluation (batch size)
        nuse = 50,

        # Tikhonov regularization weight
        λ = 1e-8,
    ),

    # Initial parameters
    p₀_fno,
    # p_fno,

    # Iterations
    100;

    # Iterations per callback
    ncallback = 1,
    callback = create_callback(fno, u_valid, t()),
)
