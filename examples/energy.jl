# Little LSP hack to get function signatures, go    #src
# to definition etc.                                #src
if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/ConvolutionalClosure.jl")       #src
    using .ConvolutionalClosure                     #src
end                                                 #src

# # Energy conserving latent variable model
#
# Conserve energy of augmented state.

using ConvolutionalClosure
using Expokit
using JLD2
using LinearAlgebra
using Lux
using Optimisers
using OrdinaryDiffEq
using Plots
using Printf
using Random
using SciMLSensitivity
using SparseArrays
using Zygote

Random.seed!(123)

apply_mat(u, p, t) = p * u
function solve_matrix(A, u₀, t, solver = Tsit5(); kwargs...)
    problem = ODEProblem(apply_mat, u₀, extrema(t), A)
    # problem = ODEProblem(DiffEqArrayOperator(A), u₀, extrema(t))
    solve(problem, solver; saveat = t, kwargs...)
end

l() = 1.0

s = 10
M = 50
N = s * M

output = "output/energy/N$(N)_M$(M)/"
mkpath(output)

y = LinRange(0, l(), N + 1)[2:end]
x = LinRange(0, l(), M + 1)[2:end]

# FN = circulant(N, [-1, 1], -[-N / 2, N / 2])
# FM = circulant(M, [-1, 1], -[-M / 2, M / 2])
FN = circulant(N, -3:3, N / l() * [1, -9, 45, 0, -45, 9, -1] / 60)
FM = circulant(M, -3:3, M / l() * [1, -9, 45, 0, -45, 9, -1] / 60)
plotmat(FN; title = "FN")
plotmat(FM; title = "FM")

# Filter widths
ΔΔ(x) = 5 / 100 * l() * (1 + 1 / 3 * sin(2π * x / l()))
Δ = ΔΔ.(x)
plot(x, Δ; xlabel = "x", title = "Filter width")

# Δ = 0.26 * s * l() / M
# # Δ = 0.5 * s * l() / M

# W = kron(I(M), ones(1, s)) / N
# plotmat(W)

# Discrete filter matrix
W = sum(gaussian.(Δ, x .- y' .- z .* l()) for z ∈ -2:2)
W = W ./ sum(W; dims = 2)
W[abs.(W).<1e-4] .= 0
W = W ./ sum(W; dims = 2)
W = sparse(W)
dropzeros!(W)
# W = W / sqrt(λmax) 1.0000000003
plotmat(W; title = "W")

plotmat(W .!= 0; title = "Discrete filter (sparsity)")

savefig(loc * "W.pdf")
# savefig(loc * "W_wide.png")
# savefig(loc * "W_uniform.png")

# Linear interpolant
R = W' / Matrix(W * W')
plotmat(R)

plotmat(W * W')
plotmat(W' * W)
plotmat(W * R)
plotmat(R * W)

# # Create data
#
# We will create data sets for compression, training, validation, and testing.

"""
Linear-ish frequency decay.
"""
decay(k) = 1 / (1 + abs(k))^1.2

# Maximum frequency in initial conditions
kmax = N ÷ 2

# Number of samples
n_compress = 5000
n_train = 120
n_valid = 10
n_test = 60

# Initial conditions (real-valued)
u₀ = create_data(y, kmax, 1; decay)[:]
u₀_compress = create_data(y, kmax, n_compress; decay)
u₀_train = create_data(y, kmax, n_train; decay)
u₀_valid = create_data(y, kmax, n_valid; decay)
u₀_test = create_data(y, kmax, n_test; decay)
plot(y, u₀_train[:, 1:3]; xlabel = "x", label = ["u₁" "u₂" "u₃"], title = "Data samples")

savefig(loc * "data_samples.pdf")

#   a b c d e
# 1 1 1 1 1 1 1 1 1 1 1 1
#       1       1       1
#
#           a b c d e
# 1 1 1 1 1 1 1 1 1 1 1 1
#       1       1       1
#
# d e               a b c
# 1 1 1 1 1 1 1 1 1 1 1 1
#       1       1       1

plot(; xlabel = "x", title = "Local filter kernels")
plot!(y, W[10, :]; label = @sprintf "x = %.2f" x[10])
plot!(y, W[20, :]; label = @sprintf "x = %.2f" x[20])
plot!(y, W[30, :]; label = @sprintf "x = %.2f" x[30])
plot!(y, W[40, :]; label = @sprintf "x = %.2f" x[40])
plot!(y, W[50, :]; label = @sprintf "x = %.2f" x[50])

maximum(i -> length(W[i, :].nzval), 1:M)

scatter(y, W[11, :])

# Energy compression
vals, vecs = eigen(Matrix(I - s * W'W))

scatter(vals; label = false, color = map(v -> v > 0 ? 1 : 2, vals), title = "Eigenvalues")

savefig(loc * "eigenvalues.pdf")
# savefig(loc * "eigenvalues_wide.pdf")
# savefig(loc * "eigenvalues_uniform.pdf")

plotmat(I - s * W'W)
plotmat(W'W)

i = 55:75
plot(
    y,
    vecs[:, i];
    # label = i',
    xlabel = "x",
    label = false,
    title = "Eigenvectors $i of I/N - W'W/M",
)

plot!(y, 0.4ΔΔ.(y) .- 0.085)

plot(
    (plot(
        y,
        vecs[:, i];
        # label = i',
        label = false,
        xticks = false,
        yticks = false,
        # xlabel = "x",
        title = i,
    ) for i = [1:7; 49; 50; 51; 52; 500])...,
    # layout = (2, 1),
    # title = "Eigenvectors of I/N - W'W/M",
    # size = (1200, 800),
    size = (800, 500),
)

plot(x, W * vecs[:, 500])

savefig(loc * "eigenvectors.pdf")
# savefig(loc * "eigenvectors_wide.pdf")
# savefig(loc * "eigenvectors_uniform.pdf")

w = vecs * Diagonal(vecs' * u₀)

i = 1:50
plot(y, u₀)
plot!(x, W * u₀)
plot!(y, sum(w[:, i]; dims = 2))
plot!(y, u₀ - sum(w[:, i]; dims = 2))

vals
r = 1
TT = sqrt(r) .* sqrt.(vals[1:r]) .* vecs[:, 1:r]'

T = apply_stencils_nonsquare

function loss_comp(p, u)
    v = W * u
    w = T(u, p)
    Ev = v .* v
    Ew = w .* w
    Eu = u .* u
    sum(abs2, sum(Ev; dims = 1) ./ M + sum(Ew; dims = 1) ./ M - sum(Eu; dims = 1) ./ N) /
    size(u, 2)
    # sum(abs2, eachcol(Ev + Ew - Eu)) / size(u, 2)
end

function loss_comp(p; u = u₀_compress, nuse = 100)
    nsample = size(u, 2)
    i = Zygote.@ignore sort(shuffle(1:nsample)[1:nuse])
    loss_comp(p, u[:, i])
end

E(u) = u'u / 2 * l() / length(u)

d = 21
# τ₀ = fill(1 / d, M, d)
# τ₀ = zeros(M, d)
τ₀ = 0.01 * randn(M, d)

τ = τ₀
loss_comp(τ₀; u = u₀_compress[:, 1:100], nuse = 100)
loss_comp(τ; u = u₀_compress[:, 1:100], nuse = 100)
first(gradient(loss_comp, τ))

τ = train(
    loss_comp,
    τ₀,
    # p,
    5000;
    opt = Optimisers.ADAM(0.001),
    callback = (i, τ) -> println(
        "Iteration \t $i, loss: $(loss_comp(τ; u = u₀_compress[:, 1:100], nuse = 100))",
    ),
    ncallback = 10,
)

v₀ = W * u₀
w₀ = T(u₀, τ)
q₀ = [v₀; w₀]

plot(; title = "Initial conditions", xlabel = "x")
plot!(y, u₀; label = "u")
plot!(x, v₀; label = "Wu")
plot!(x, w₀; label = "Tu")
# plot!(x, v₀ + w₀; label = "v + w")

savefig(loc * "initial_conditions_energy.pdf")

plot(; xlabel = "x")
plot!(y, u₀ .^ 2 ./ 2; label = "e(u)")
plot!(x, v₀ .^ 2 ./ 2; label = "e(v)")
plot!(x, w₀ .^ 2 ./ 2; label = "e(w)")
# plot!(x, v₀ .^ 2 ./ 2 .+ w₀ .^ 2 ./ 2; label = "e(v) + e(w)")

# Evaluation times
t = LinRange(0.0, 1.0, 501)
t_train = LinRange(0, 0.1, 41)
t_valid = LinRange(0, 0.2, 26)
t_test = LinRange(0, 1.0, 61)

# Unfiltered solutions
u = solve_matrix(FN, u₀, t; reltol = 1e-4, abstol = 1e-6)
u_train = solve_matrix(FN, u₀_train, t_train; reltol = 1e-4, abstol = 1e-6)
u_valid = solve_matrix(FN, u₀_valid, t_valid; reltol = 1e-4, abstol = 1e-6)
u_test = solve_matrix(FN, u₀_test, t_test; reltol = 1e-4, abstol = 1e-6)

# Filtered solutions
v = apply_filter(W, u)
v_train = apply_filter(W, u_train)
v_valid = apply_filter(W, u_valid)
v_test = apply_filter(W, u_test)

# Latent solutions
w = reshape(T(Array(u), τ), M, :)
w_train = reshape(T(reshape(Array(u_train), N, :), τ), M, n_train, :)
w_valid = reshape(T(reshape(Array(u_valid), N, :), τ), M, n_valid, :)
w_test = reshape(T(reshape(Array(u_test), N, :), τ), M, n_test, :)

# Augmented states
q = [v; w]
q_train = [v_train; w_train]
q_valid = [v_valid; w_valid]
q_test = [v_test; w_test]

# Time derivatives
dudt = apply_filter(FN, u)
dudt_train = apply_filter(FN, u_train)
dudt_valid = apply_filter(FN, u_valid)
dudt_test = apply_filter(FN, u_test)

# Filtered time derivatives
dvdt = apply_filter(W, dudt)
dvdt_train = apply_filter(W, dudt_train)
dvdt_valid = apply_filter(W, dudt_valid)
dvdt_test = apply_filter(W, dudt_test)

# Filtered latent time derivatives
dwdt = reshape(T(Array(dudt), τ), M, :)
dwdt_train = reshape(T(reshape(Array(dudt_train), N, :), τ), M, n_train, :)
dwdt_valid = reshape(T(reshape(Array(dudt_valid), N, :), τ), M, n_valid, :)
dwdt_test = reshape(T(reshape(Array(dudt_test), N, :), τ), M, n_test, :)

# Augmented time derivatives
dqdt = [dvdt; dwdt]
dqdt_train = [dvdt_train; dwdt_train]
dqdt_valid = [dvdt_valid; dwdt_valid]
dqdt_test = [dvdt_test; dwdt_test]

decomp = svd(Matrix(W))

UU, SS, VV = decomp

plotmat(R)
plotmat(VV*Diagonal(1 ./ SS) * UU')

plotmat(R * W)

vals, vecs = eigen(I - VV * VV')
scatter(vals)


plot(; xlabel = "x")
plot!(y, u₀)
plot!(y, R * W * u₀)
plot!(y, (I - VV * VV') * u₀)
# plot!(y, vecs * vecs' * u₀)
plot!(y, vecs[:, M+1:end] * vecs[:, M+1:end]' * u₀)

plot(y, vecs[:, M+1:end] * vecs[:, M+1:end]' * u₀)
plot(y, vecs * vecs' * u₀)
plot(y, vecs[:, 54])

plotmat(I - VV * VV')

UU
SS
VV

plotmat(UU)
plot(y, VV[:, 1:6])

dec = svd(VV')

plotmat(dec.U)
scatter(dec.S)
plot(y, dec.V[:, 34])

plotmat(W' * UU * Diagonal(1 ./ SS.^2) * UU')

scatter(decomp.S)
scatter(vals)
scatter!(1 .- decomp.S.^2 .* 10)

r = 50
TT = vecs[:, 1:r] * Diagonal(vals[1:r]) * vecs[:, 1:r]'
plotmat(TT)

plot(; xlabel = "x")
plot!(y, u₀; label = "u")
plot!(y, s * W' * v₀; label = "Rv")
# plot!(y, TT * u₀; label = "T'Tu")
plot!(y, s * W' * v₀ + TT * u₀; label = "Rv + T'Tu")

Ew = sum(u .* (TT * u); dims = 1)[:] / 2N
r

plot(; title = "Kinetic energy", xlabel = "t", legend = :right)
plot!(t, E.(eachcol(u)); label = "E(u)")
plot!(t, E.(eachcol(v)); label = "E(Wu)")
# plot!(t, E.(eachcol(v)) + Ew; label = "E(Wu) + E(Tu)")
plot!(t, E.(eachcol(v)) .+ E.(eachcol(w)); label = "E(Wu) + E(Tu)")
# ylims!((0, ylims()[2]))

savefig(loc * "energy_compression.pdf")

# Neural networks for K
# Sice K is square, K^T is given by K(reverse(p))
r11, r12, r22 = (3, 3, 3)
d11, d12, d22 = (2 * r11 + 1, 2 * r12 + 1, 2 * r22 + 1)

p₀11 = 0.01 * randn(M, d11)
p₀12 = 0.01 * randn(M, d12)
p₀22 = 0.01 * randn(M, d22)

# This should be zero
v₀' * (apply_stencils(v₀, p₀11) - apply_stencils_transpose(v₀, p₀11))
v₀' * (apply_stencils(v₀, p₀12) - apply_stencils_transpose(v₀, p₀12))
v₀' * (apply_stencils(v₀, p₀22) - apply_stencils_transpose(v₀, p₀22))

function Q(q, p, t)
    p11 = p[:, 1:d11]
    p12 = p[:, d11+1:d11+d12]
    p22 = p[:, d11+d12+1:end]
    u = q[1:M, :]
    w = q[M+1:end, :]
    duu = apply_stencils(u, p11) - apply_stencils_transpose(u, p11)
    duw = apply_stencils(w, p12)
    dwu = -apply_stencils_transpose(u, p12)
    dww = apply_stencils(w, p22) - apply_stencils_transpose(w, p22)
    [duu + duw; dwu + dww]
end

# Callback for studying convergence
function create_callback(Q, q, t; iplot = 1:10)
    v = q[1:M, iplot, :]
    # w = q[M+1:end, iplot, :]
    hist_i = Int[]
    hist_err = zeros(0)
    err_nomodel =
        relerr(Array(solve_matrix(FM, v[:, :, 1], t; reltol = 1e-4, abstol = 1e-6)), v)
    function callback(i, p)
        sol = solve_equation(Q, q[:, iplot, 1], p, t; reltol = 1e-4, abstol = 1e-6)
        vpred = sol[1:M, :, :]
        err = relerr(vpred, v)
        println("Iteration $i \t average relative error $err")
        push!(hist_i, i)
        push!(hist_err, err)
        pl = plot(; title = "Average relative error", xlabel = "Iterations")
        hline!(pl, [err_nomodel]; color = 1, linestyle = :dash, label = "No closure")
        plot!(pl, hist_i, hist_err; label = "With closure")
        display(pl)
    end
end

q₀ = [v₀; w₀]
p₀ = [p₀11 p₀12 p₀22]
size(p₀)

repeat(q₀, 1, 5)
Q(q₀, p₀, 0.0)

@time gradient(p -> sum(Q(repeat(q₀, 1, 100), p, 0.0)), p₀);

derivative_loss(
    Q,
    p₀,
    reshape(dqdt_train, 2M, :),
    reshape(q_train, 2M, :);
    nuse = 100,
    λ = 1e-8,
)

first(
    gradient(
        p -> derivative_loss(
            Q,
            p,
            reshape(dqdt_train, 2M, :),
            reshape(q_train, 2M, :);
            nuse = 100,
            λ = 1e-8,
        ),
        p₀,
    ),
)

create_callback(Q, q_valid, t_valid)
create_callback(Q, q_valid, t_valid)(0, p₀)

p = train(
    # Loss
    p -> derivative_loss(
        Q,
        p,

        # Merge solution and time dimension, with new size `(ncomponent, nsolution*ntime)`
        reshape(dqdt_train, 2M, :),
        reshape(q_train, 2M, :);

        # Number of random data samples for each loss evaluation (batch size)
        nuse = 200,

        # Tikhonov regularization weight
        λ = 1e-8,
    ),

    # Initial parameters
    p₀,
    # p,

    # Number of iterations
    5_000;

    # Optimiser
    opt = Optimisers.ADAM(0.001),

    # Callback
    callback = (cb = create_callback(Q, q_valid, t_valid); cb),
    # callback = (i, p) -> cb(i + 27_500, p),
    # callback = (i, p) -> println("Iteration $i"),

    # Number of iterations between callbacks
    ncallback = 10,
)

p11 = p[:, 1:d11]
p12 = p[:, d11+1:d11+d12]
p22 = p[:, d11+d12+1:end]

sum(p11; dims = 1) / M

k11 = p11 - transpose_kernel(p11)
k12 = p12
k22 = p22 - transpose_kernel(p22)

# Sixth order convection stencil
pref6 = M / l() * [0, 0, 0, 0, -45, 9, -1]' / 60
kref6 = M / l() * [1, -9, 45, 0, -45, 9, -1]' / 60

# Fourth order convection stencil
pref4 = M / l() * [0, 0, 0, 0, -8, 1, 0]' / 12
kref4 = M / l() * [0, -1, 8, 0, -8, 1, 0]' / 12

# Second order convection stencil
pref2 = M / l() * [0, 0, 0, 0, -1, 0, 0]' / 2
kref2 = M / l() * [0, 0, 1, 0, -1, 0, 0]' / 2

[sum(k11; dims = 1) / M; kref2]
[sum(p11; dims = 1) / M; pref2]

plot(; xlabel = "Offset index", title = "Convection stencil")
bar!(-r11:r11, kref6[:] / M; label = "Coarse discretization (6th order)")
bar!(-r11:r11, sum(k11; dims = 1)[:] / M / M; label = "Average Learned")
bar!(-r11:r11, kref2[:] / M; label = "Coarse discretization (2nd order)")
scatter!(-r11:r11, k11' / M; label = false, color = 2, opacity = 0.2)

savefig(loc * "convection_stencil.pdf")

# p = pref

cb(20_001, p)
cb(20_004, pref)


P = [k11 zeros(M, M - 7)]
P = reduce(vcat, circshift(P[[4 + i], :], (0, i)) for i = -3:M-4)
plotmat(P; title = "K11")

savefig(loc * "K11.pdf");

kreff6 = repeat(kref6, M)
Pref = [kreff6 zeros(M, M - 7)]
Pref = reduce(vcat, circshift(Pref[[4 + i], :], (0, i)) for i = -3:M-4)

plotmat(P)
plotmat(Pref)

k12
P12 = [k12 zeros(M, M - 7)]
P12 = reduce(vcat, circshift(P12[[4 + i], :], (0, i)) for i = -3:M-4)
plotmat(P12; title = "K12")

savefig(loc * "K12.pdf");

k22
P22 = [k22 zeros(M, M - 7)]
P22 = reduce(vcat, circshift(P22[[4 + i], :], (0, i)) for i = -3:M-4)
plotmat(P22; title = "K22")

savefig(loc * "K22.pdf");

qmodel = solve_equation(Q, reshape(q₀, :, 1), p, t; reltol = 1e-4, abstol = 1e-6)
vmodel = qmodel[1:M, 1, :]
wmodel = qmodel[M+1:end, 1, :]

plot(; title = "Kinetic energy", xlabel = "t", legend = :right)
plot!(t, E.(eachcol(u)); label = "E(u)")
plot!(t, E.(eachcol(v)); label = "E(Wu)")
# plot!(t, E.(eachcol(vmodel)); label = "E(qv)")
plot!(t, E.(eachcol(v)) .+ E.(eachcol(w)); label = "E(Wu) + E(Tu)")
plot!(
    t,
    E.(eachcol(vmodel)) .+ E.(eachcol(wmodel));
    label = "E(q) for all K",
    color = 3,
    linestyle = :dash,
)
# ylims!((0, ylims()[2]))

savefig(loc * "energy_compression_q.pdf")

# Plot example solution
for (i, t) ∈ collect(enumerate(t))#[1:200]
    pl = plot(; xlabel = "x", title = @sprintf("t = %.2f", t), ylims = extrema(u[:, :]))
    plot!(pl, y, u[i]; label = "Unfiltered")
    plot!(pl, x, v[:, i]; label = "Filtered exact")
    # plot!(pl, x, vmodel[:, i]; label = "Filtered")
    plot!(pl, x, w[:, i]; label = "Sub exact")
    # scatter!(pl, x, wmodel[:, i]; label = "Sub")
    display(pl)
    sleep(0.05) # Time for plot pane to update
end

filename = joinpath(output, "stencils.jld2")
# jldsave(filename; τ, p11, p12, p22)
# τ, p11, p12, p22 = load(filename, "τ", "p11", "p12", "p22")
# p = [p11 p12 p22]
