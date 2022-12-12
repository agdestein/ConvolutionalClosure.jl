if isdefined(@__MODULE__, :LanguageServer)
    include("../src/ConvolutionalClosure.jl")
    using .ConvolutionalClosure
end

using ConvolutionalClosure
using Expokit
# using Integrals
using LinearAlgebra
using Lux
using OrdinaryDiffEq
using Plots
using SparseArrays

apply_mat(u, p, t) = p * u
function solve_matrix(A, u₀, t; kwargs...)
    problem = ODEProblem(apply_mat, u₀, extrema(t), A)
    solve(problem, Tsit5(); saveat = t, kwargs...)
end

l() = 1.0

N = 400
M = 100

loc = "output/N$(N)_M$(M)/"
mkpath(loc)

ξ = LinRange(0, l(), N + 1)[2:end]
x = LinRange(0, l(), M + 1)[2:end]

FN = circulant(N, [-1, 1], [-N / 2, N / 2])
FM = circulant(M, [-1, 1], [-M / 2, M / 2])
plotmat(FN; title = "DN")
plotmat(FM; title = "DM")

# Filter widths
ΔΔ(x) = (1 + 1 / 3 * sin(2π * x / l())) * 3 * l() / M
Δ = ΔΔ.(x)
plot(x, Δ; xlabel = "x", title = "Filter width")

# # Filter width
# Δ = 4 * l() / M

# Discrete filter matrix
W = sum(gaussian.(Δ, x .- ξ' .- z .* l()) for z ∈ -2:2)
W = W ./ sum(W; dims = 2)
W[abs.(W).<1e-6] .= 0
W = W ./ sum(W; dims = 2)
W = sparse(W)
dropzeros!(W)
plotmat(W; title = "Discrete filter")
plotmat(W .!= 0; title = "Discrete filter (sparsity)")

# Interpolant
P = interpolation_matrix(l(), x, ξ)
plotmat(P)

# Full reconstructor
P = Matrix(W'W + 1e-4 * I) \ Matrix(W')
P = Matrix(W') / Matrix(W * W' + 1e-4 * I)
plotmat(P)
plotmat(P * W)
plotmat(W * P)

# Mori-Zwanzig matrices
A = W * FN * P
B = W * FN
C = (I - P * W) * FN * P
D = (I - P * W) * FN

plotmat(A)
plotmat(B)
plotmat(C)
plotmat(D)

plotmat(P * W)

savefig(loc * "PW.png")


"""
Linear-ish frequency decay.
"""
decay(k) = 1 / (1 + abs(k))^1.2

# Initial conditions
u₀ = create_data(ξ, N ÷ 2, 1; decay)[:]
for i = 1:5
    u₀ = P * W * u₀
end

pl = plot(; xlabel = "x", title = "Initial conditions")
plot!(ξ, u₀; label = "Unfiltered")
plot!(x, W * u₀; label = "Filtered")
pl

savefig(loc * "initial_conditions.png")

# Evaluation time
t = LinRange(0, 1.0, 501)

# Unfiltered solution
u = solve_matrix(FN, u₀, t; reltol = 1e-8, abstol = 1e-10)

# Filtered solution
ū = W * u
e = (I - P * W) * u

plot(x, ū[:, 1]; label = "ū")
plot!(ξ, e[:, 1]; label = "u'")

# Filtered time derivatives
dūdt = W * FN * u

# Individual terms in dūdt
markov = A * ū
noise = mapreduce(t -> B * expmv(t, D, e[:, 1]), hcat, t)
memory = dūdt - markov - noise

pl = plot(; xlabel = "t", title = "Term size (dū/dt)")
plot!(t, norm.(eachcol(markov)); label = "Markov")
plot!(t, norm.(eachcol(noise)); label = "Noise")
plot!(t, norm.(eachcol(memory)); label = "Memory")
pl

savefig(article * "termsize_ubar.pdf")
savefig(loc * "termsize_ubar.png")

# solve(
#     IntegralProblem((s, _) -> B * D * exp(D * (t - s)) * C * W * u(t), 0, t),
#     HCubatureJL();
#     reltol = 1e-4,
#     abstol = 1e-6,
# )

Δt = t[2] - t[1]
edt = [B * D * exp(Matrix(D) * t) * C for t ∈ t]
γ_i(i, j) = edt[i-j+1] * ū[:, j]
γ = hcat(
    zeros(M),
    reduce(
        hcat,
        sum((γ_i(i, j) + γ_i(i, j + 1)) * Δt / 2 for j = 1:i-1) for i = 2:length(t)
    ),
)

dnoisedt = mapreduce(t -> B * D * expmv(t, D, e[:, 1]), hcat, t)

pl = plot(; xlabel = "t", title = "Term size (dw/dt)")
plot!(t, norm.(eachcol(B * C * ū)); label = "B C ū")
plot!(t, norm.(eachcol(dnoisedt)); label = "B D e^Dt u₀'")
plot!(t, norm.(eachcol(γ)); label = "γ")
pl

savefig(article * "termsize_w.pdf")
savefig(loc * "termsize_w.png")

pl = plot()
plot!(t, norm.(eachcol(γ + B * C * ū)))
plot!(
    (t[1:end-1] + t[2:end]) / 2,
    norm.(eachcol((memory[:, 2:end] - memory[:, 1:end-1]) / Δt)),
)
pl

Δt = t[2] - t[1]
edt = [B * exp(Matrix(D) * t) * C for t ∈ t]
γ_i(i, j) = edt[i-j+1] * ū[:, j]
γ = hcat(
    zeros(M),
    reduce(
        hcat,
        sum((γ_i(i, j) + γ_i(i, j + 1)) * Δt / 2 for j = 1:i-1) for i = 2:length(t)
    ),
)

pl = plot()
plot!(t, norm.(eachcol(memory)))
plot!(t, norm.(eachcol(γ)))
pl

# Callback for studying convergence
function create_callback(f, s, t)
    ū = s[:, 1, :, :]
    iplot = 1:10
    hist_i = Int[]
    hist_err = zeros(0)
    err_nomodel = relerr(
        Array(
            solve_matrix(
                FM,
                ū[:, iplot, 1],
                t;
                reltol = 1e-4,
                abstol = 1e-6,
            ),
        ),
        ū[:, iplot, :],
    )
    err_markov = relerr(
        Array(
            solve_matrix(
                A,
                ū[:, iplot, 1],
                t;
                reltol = 1e-4,
                abstol = 1e-6,
            ),
        ),
        ū[:, iplot, :],
    )
    function callback(i, p)
        sol = solve_equation(f, s[:, :, iplot, 1], p, t; reltol = 1e-4, abstol = 1e-6)
        v = sol[:, 1, :, :]
        err = relerr(v, ū[:, iplot, :])
        println("Iteration $i \t average relative error $err")
        push!(hist_i, i)
        push!(hist_err, err)
        pl = plot(; title = "Average relative error", xlabel = "Iterations")
        hline!(pl, [err_nomodel]; color = 1, linestyle = :dash, label = "dv/dt = Fₘ v")
        hline!(pl, [err_markov]; color = 2, linestyle = :dash, label = "dv/dt = A v")
        plot!(pl, hist_i, hist_err; label = "dv/dt = A v + w")
        display(pl)
    end
end

# Initialize NN
p₀, closure = convolutional_matrix_closure(
    # Kernel radii (nlayer)
    [5, 5, 3],

    # Number of channels (nlayer + 1)
    # First is number of input channels, last must be 1
    [3, 4, 3, 1],

    # Activation functions (nlayer)
    [Lux.relu, Lux.relu, identity],

    # Bias
    [true, true, true];

    # Input channels
    channel_augmenter = s -> hcat(
        # Vanilla channels s = [v w]
        s,

        # Filter width channel for non-uniformity (same for each batch)
        repeat(Δ, 1, 1, size(s, 3)),
    ),
)

state = hcat(W * u₀, zeros(M))
state = repeat(state, 1, 1, 5)
closure(state, p₀, 0)

function system(state, p, t)
    nsample = size(state, 3)
    v, s = eachslice(state; dims = 2)
    dv = A * v + s
    # dn = B * (D * (exp(Matrix(D) * t) * e)
    ds = B * (C * v) + closure(state, p, t)
    hcat(reshape(dv, :, 1, nsample), reshape(ds, :, 1, nsample))
end
system(state::AbstractMatrix, p, t) = reshape(system(reshape(state, :, 2, 1), p, t), :, 2)

system(state, p₀, 0)

# Maximum frequency in initial conditions
K = N ÷ 2

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
t_train = LinRange(0, 0.2, 41)
t_valid = LinRange(0, 0.5, 26)
t_test = LinRange(0, 1.0, 61)

# Unfiltered solutions
u_train = solve_matrix(FN, u₀_train, t_train; reltol = 1e-4, abstol = 1e-6)
u_valid = solve_matrix(FN, u₀_valid, t_valid; reltol = 1e-4, abstol = 1e-6)
u_test = solve_matrix(FN, u₀_test, t_test; reltol = 1e-4, abstol = 1e-6)

# Filtered solutions
ū_train = apply_filter(W, u_train)
ū_valid = apply_filter(W, u_valid)
ū_test = apply_filter(W, u_test)

# Sub-filter solutions
e_train = apply_filter(I - P * W, u_train)
e_valid = apply_filter(I - P * W, u_valid)
e_test = apply_filter(I - P * W, u_test)

# Filtered time derivatives (for derivative fitting)
dūdt_train = apply_filter(W, apply_filter(FN, u_train))
dūdt_valid = apply_filter(W, apply_filter(FN, u_valid))
dūdt_test = apply_filter(W, apply_filter(FN, u_test))

# Markovian terms
markov_train = apply_filter(A, ū_train)
markov_valid = apply_filter(A, ū_valid)
markov_test = apply_filter(A, ū_test)

# Noise terms
noise_train = reshape(
    mapreduce(t -> B * exp(Matrix(D) * t) * e_train[:, :, 1], hcat, t_train),
    M,
    n_train,
    :,
)
noise_valid = reshape(
    mapreduce(t -> B * exp(Matrix(D) * t) * e_valid[:, :, 1], hcat, t_valid),
    M,
    n_valid,
    :,
)
noise_test = reshape(
    mapreduce(t -> B * exp(Matrix(D) * t) * e_test[:, :, 1], hcat, t_test),
    M,
    n_test,
    :,
)

# Latent sub-filter variables
w_train = dūdt_train - markov_train
w_valid = dūdt_valid - markov_valid
w_test = dūdt_test - markov_test

# Latent time derivatives
dwdt_train = apply_filter(W * FN^2 - A * W * FN, u_train)
dwdt_valid = apply_filter(W * FN^2 - A * W * FN, u_valid)
dwdt_test = apply_filter(W * FN^2 - A * W * FN, u_test)

# States (ū, w)
s_train = reshape(vcat(ū_train, w_train), M, 2, n_train, :)
s_valid = reshape(vcat(ū_valid, w_valid), M, 2, n_valid, :)
s_test = reshape(vcat(ū_test, w_test), M, 2, n_test, :)

# Time derivatives of states
dsdt_train = reshape(vcat(dūdt_train, dwdt_train), M, 2, n_train, :)
dsdt_valid = reshape(vcat(dūdt_valid, dwdt_valid), M, 2, n_valid, :)
dsdt_test = reshape(vcat(dūdt_test, dwdt_test), M, 2, n_test, :)

callback = create_callback(system, s_valid, t_valid)
callback(1, p₀)
callback(2, p₀)

p_df = train(
    p -> derivative_loss(
        system,
        p,

        # Merge solution and time dimension, with new size `(nx, nsolution*ntime)`
        reshape(dsdt_train, M, 2, :),
        reshape(s_train, M, 2, :);

        # Number of random data samples for each loss evaluation (batch size)
        nuse = 200,

        # Tikhonov regularization weight
        λ = 1e-8,
    ),
    # p₀,
    p_df,
    1000;
    ncallback = 100,
    callback = create_callback(system, s_valid, t_valid),
)

savefig(loc * "loss.png")
