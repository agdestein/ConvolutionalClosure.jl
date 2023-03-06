if isdefined(@__MODULE__, :LanguageServer)
    include("../src/ConvolutionalClosure.jl")
    using .ConvolutionalClosure
end

using ConvolutionalClosure
using Expokit
using LinearAlgebra
using Lux
using OrdinaryDiffEq
using Plots
using SciMLSensitivity
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

y = LinRange(0, l(), N + 1)[2:end]
x = LinRange(0, l(), M + 1)[2:end]

FN = circulant(N, [-1, 1], [-N / 2, N / 2])
FM = circulant(M, [-1, 1], [-M / 2, M / 2])
plotmat(FN; title = "FN")
plotmat(FM; title = "FM")

# # Filter widths
# # ΔΔ(x) = (1 + 1 / 2 * sin(2π * x / l())) * 3 * l() / M
# ΔΔ(x) = (1 + 1 / 3 * sin(2π * x / l())) * 3 * l() / M
# Δ = ΔΔ.(x)
# plot(x, Δ; xlabel = "x", title = "Filter width")

# Filter width
Δ = 4 * l() / M

# Discrete filter matrix
W = sum(gaussian.(Δ, x .- y' .- z .* l()) for z ∈ -2:2)
W = W ./ sum(W; dims = 2)
W[abs.(W).<1e-6] .= 0
W = W ./ sum(W; dims = 2)
W = sparse(W)
dropzeros!(W)
plotmat(W; title = "Discrete filter")
plotmat(W .!= 0; title = "Discrete filter (sparsity)")

# # Piece-wise constant interpolant
# R = constant_interpolator(l(), x, y)
# plotmat(R)

# Linear interpolant
R = linear_interpolator(l(), x, y)
plotmat(R)

# # Full reconstructor
# R = Matrix(W') / Matrix(W * W' + 1e-4 * I)

plotmat(R)
plotmat(R * W)
plotmat(W * R)

# Mori-Zwanzig matrices
A = W * FN * R
B = W * FN
C = (I - R * W) * FN * R
D = (I - R * W) * FN

plotmat(A)
plotmat(B)
plotmat(C)
plotmat(D)

plotmat(R * W)
Matrix(R * W)

Matrix(R'R)

savefig(loc * "RW.png")

"""
Linear-ish frequency decay.
"""
decay(k) = 1 / (1 + abs(k))^1.2

# Initial conditions
u₀ = create_data(y, N ÷ 2, 1; decay)[:]
# for i = 1:5
#     u₀ = R * W * u₀
# end

pl = plot(; xlabel = "x", title = "Example signal")
plot!(y, u₀; label = "u")
# scatter!(x, W * u₀; label = "Filtered")
plot!(y, R * W * u₀; label = "RWu")
plot!(y, P * V * u₀; label = "PVu")
pl

savefig(loc * "example_signal.pdf")

# Evaluation time
t = LinRange(0, 0.5, 501)

# Unfiltered solution
u = solve_matrix(FN, u₀, t; reltol = 1e-8, abstol = 1e-10)

# Filtered solution
ū = W * u

# Unresolved solution
e = V * u

# Filtered time derivatives
dūdt = W * FN * u

# Kinetic energy
E(u) = 1 / 2 * l() / length(u) * u'u
E = u -> 1 / 2 * l() / length(u) * u'u
pl = plot(; xlabel = "t", title = "Kinetic energy")
plot!(t, map(E, eachcol(u)); label = "E(u)")
plot!(t, map(E, eachcol(R * ū)); label = "E(ū)")
plot!(t, E.(eachcol(R * ū)) + E.(eachcol(P * e)); label = "E(ū) + E(u')")
# ylims!((0, ylims()[2]))
pl

# savefig(loc * "kinetic_energy.pdf")

# Individual terms in dūdt
markov = A * ū
noise = mapreduce(t -> B * expmv(t, D, e[:, 1]), hcat, t)
memory = dūdt - markov - noise

kernel = [norm(B * expmv(t[end] - t[i], D, C * W * u[i])) for i = 1:length(u)]

plot(t, kernel; xlabel = "s", title = "Memory kernel, t = $(t[end])", legend = false)
ylims!((0, ylims()[2]))
# savefig(loc * "kernel.pdf")

pl_Δ = plot(
    Δ,
    x;
    ylabel = "x",
    ylims = (0, l()),
    label = false,
    title = "Δ(x)",
    xticks = [minimum(Δ), sum(extrema(Δ)) / 2, maximum(Δ)],
    # xlims = (0, maximum(Δ)),
)
pl_ū = plotsol(x, t, ū; title = "Filtered solution")#, ylabel = "")
pl_markov = plotsol(x, t, markov; title = "Markovian term")#, ylabel = "")
pl_noise = plotsol(x, t, noise; title = "Noise term")#, ylabel = "")
pl_memory = plotsol(x, t, memory; title = "Memory term")#, ylabel = "")

plot(pl_Δ, pl_ū; layout = grid(1, 2, widths=[0.3 ,0.7]))
savefig(loc * "filtered.png")
savefig(loc * "filtered.pdf")
plot(pl_Δ, pl_markov; layout = grid(1, 2, widths=[0.3 ,0.7]))
savefig(loc * "markov.png")
savefig(loc * "markov.pdf")
plot(pl_Δ, pl_noise; layout = grid(1, 2, widths=[0.3 ,0.7]))
savefig(loc * "noise.png")
savefig(loc * "noise.pdf")
plot(pl_Δ, pl_memory; layout = grid(1, 2, widths=[0.3 ,0.7]))
savefig(loc * "memory.png")
savefig(loc * "memory.pdf")

pl = plot(; legend = :right, xlabel = "t", title = "Term size (dū/dt)")
plot!(t, norm.(eachcol(markov)); label = "Markov")
plot!(t, norm.(eachcol(noise)); label = "Noise")
plot!(t, norm.(eachcol(memory)); label = "Memory")
pl

savefig(article * "termsize_ubar.pdf")
savefig(loc * "termsize_ubar.pdf")

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
function create_callback(f, s, t, p₀)
    ū = s[:, 1, :, :]
    iplot = 1:10
    hist_i = Int[]
    hist_err = zeros(0)
    err_nomodel = relerr(
        Array(solve_matrix(FM, ū[:, iplot, 1], t; reltol = 1e-4, abstol = 1e-6)),
        ū[:, iplot, :],
    )
    err_markov = relerr(
        Array(solve_matrix(A, ū[:, iplot, 1], t; reltol = 1e-4, abstol = 1e-6)),
        ū[:, iplot, :],
    )
    err_w = relerr(
        solve_equation(f, s[:, :, iplot, 1], zero(p₀), t; reltol = 1e-4, abstol = 1e-6)[
            :,
            1,
            :,
            :,
        ],
        ū[:, iplot, :],
    )
    function callback(i, p)
        sol = solve_equation(f, s[:, :, iplot, 1], p, t; reltol = 1e-4, abstol = 1e-6)
        v = sol[:, 1, :, :]
        err = relerr(v, ū[:, iplot, :])
        println("Iteration $i \t average relative error $err")
        push!(hist_i, i)
        push!(hist_err, err)
        pl = plot(;
            title = "Average relative error ||v - ū|| / ||ū||",
            xlabel = "Iterations",
        )
        hline!(pl, [err_nomodel]; color = 1, linestyle = :dash, label = "dv/dt = Fₘv")
        hline!(pl, [err_markov]; color = 2, linestyle = :dash, label = "dv/dt = Av")
        hline!(
            pl,
            [err_w];
            color = 3,
            linestyle = :dash,
            label = "dv/dt = Av + w\ndw/dt = BCv",
        )
        plot!(pl, hist_i, hist_err; label = "dv/dt = Av + w\ndw/dt = BCv + NN(v,w)")
        display(pl)
        pl
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

function system(state, p, t)
    nsample = size(state, 3)
    v, w = eachslice(state; dims = 2)
    # n = B * (exp(Matrix(D) * t) * e)
    dv = A * v + w
    # dn = B * (D * (exp(Matrix(D) * t) * e)
    dw = B * (C * v) + closure(state, p, t)
    hcat(reshape(dv, :, 1, nsample), reshape(dw, :, 1, nsample))
end
system(state::AbstractMatrix, p, t) = reshape(system(reshape(state, :, 2, 1), p, t), :, 2)

# Maximum frequency in initial conditions
K = N ÷ 2

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

plotsol(y, t_train, u_train[:, 1, :]; title = "u")
plotsol(y, t_valid, u_valid[:, 1, :]; title = "u")
plotsol(y, t_test, u_test[:, 1, :]; title = "u")

plotsol(x, t_train, ū_train[:, 1, :]; title = "ū")
plotsol(x, t_valid, ū_valid[:, 1, :]; title = "ū")
plotsol(x, t_test, ū_test[:, 1, :]; title = "ū")

# Sub-filter solutions
e_train = apply_filter(I - R * W, u_train)
e_valid = apply_filter(I - R * W, u_valid)
e_test = apply_filter(I - R * W, u_test)

# Filtered time derivatives (for derivative fitting)
dūdt_train = apply_filter(W, apply_filter(FN, u_train))
dūdt_valid = apply_filter(W, apply_filter(FN, u_valid))
dūdt_test = apply_filter(W, apply_filter(FN, u_test))

# Markovian terms
markov_train = apply_filter(A, ū_train)
markov_valid = apply_filter(A, ū_valid)
markov_test = apply_filter(A, ū_test)

# Noise terms
o_train = reshape(
    mapreduce(t -> B * exp(Matrix(D) * t) * e_train[:, :, 1], hcat, t_train),
    M,
    n_train,
    :,
)
o_valid = reshape(
    mapreduce(t -> B * exp(Matrix(D) * t) * e_valid[:, :, 1], hcat, t_valid),
    M,
    n_valid,
    :,
)
o_test = reshape(
    mapreduce(t -> B * exp(Matrix(D) * t) * e_test[:, :, 1], hcat, t_test),
    M,
    n_test,
    :,
)

plotsol(x, t_train, o_train[:, 1, :]; title = "Noise")
plotsol(x, t_valid, o_valid[:, 1, :]; title = "Noise")
plotsol(x, t_test, o_test[:, 1, :]; title = "Noise")

# Time derivatives of noise terms
dodt_train = reshape(
    mapreduce(t -> B * D * exp(Matrix(D) * t) * e_train[:, :, 1], hcat, t_train),
    M,
    n_train,
    :,
)
dodt_valid = reshape(
    mapreduce(t -> B * D * exp(Matrix(D) * t) * e_valid[:, :, 1], hcat, t_valid),
    M,
    n_valid,
    :,
)
dodt_test = reshape(
    mapreduce(t -> B * D * exp(Matrix(D) * t) * e_test[:, :, 1], hcat, t_test),
    M,
    n_test,
    :,
)

# Latent sub-filter variables
w_train = dūdt_train - markov_train - o_train
w_valid = dūdt_valid - markov_valid - o_valid
w_test = dūdt_test - markov_test - o_test

plotsol(x, t_train, w_train[:, 1, :]; title = "w")
plotsol(x, t_valid, w_valid[:, 1, :]; title = "w")
plotsol(x, t_test, w_test[:, 1, :]; title = "w")

# Latent time derivatives
dwdt_train = apply_filter(W * FN^2 - A * W * FN, u_train) - dodt_train
dwdt_valid = apply_filter(W * FN^2 - A * W * FN, u_valid) - dodt_valid
dwdt_test = apply_filter(W * FN^2 - A * W * FN, u_test) - dodt_test

# States s = (ū, w)
s_train = reshape(vcat(ū_train, w_train), M, 2, n_train, :)
s_valid = reshape(vcat(ū_valid, w_valid), M, 2, n_valid, :)
s_test = reshape(vcat(ū_test, w_test), M, 2, n_test, :)

# Time derivatives of states
dsdt_train = reshape(vcat(dūdt_train, dwdt_train), M, 2, n_train, :)
dsdt_valid = reshape(vcat(dūdt_valid, dwdt_valid), M, 2, n_valid, :)
dsdt_test = reshape(vcat(dūdt_test, dwdt_test), M, 2, n_test, :)

callback = create_callback(system, s_valid, t_valid, p₀)
callback(1, p₀)
callback(2, p₀)

p_df = train(
    p -> derivative_loss(
        system,
        p,

        # Merge solution and time dimension, with new size `(nx, 2, nsolution*ntime)`
        reshape(dsdt_train, M, 2, :),
        reshape(s_train, M, 2, :);

        # Number of random data samples for each loss evaluation (batch size)
        nuse = 100,

        # Tikhonov regularization weight
        λ = 1e-8,
    ),
    p₀,
    # p_df,
    5000;
    ncallback = 100,
    callback = create_callback(system, s_valid, t_valid, p₀),
)

savefig(loc * "loss_df.png")

p_tf = train(
    p -> trajectory_loss(
        system,
        p,
        s_train,
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
    p₀,
    # p_tf,
    100;
    ncallback = 10,
    callback = create_callback(system, s_valid, t_valid, p₀),
)

savefig(loc * "loss_tf.png")
