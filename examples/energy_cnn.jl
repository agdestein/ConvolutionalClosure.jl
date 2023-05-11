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

loc = "output/expansion/N$(N)_M$(M)/"
mkpath(loc)

ξ = LinRange(0, l(), N + 1)[2:end]
x = LinRange(0, l(), M + 1)[2:end]

# FN = circulant(N, [-1, 1], -[-N / 2, N / 2])
# FM = circulant(M, [-1, 1], -[-M / 2, M / 2])
FN = circulant(N, -3:3, N / l() * [1, -9, 45, 0, -45, 9, -1] / 60)
FM = circulant(M, -3:3, M / l() * [1, -9, 45, 0, -45, 9, -1] / 60)
plotmat(FN; title = "FN")
plotmat(FM; title = "FM")

# Discrete filter matrix
Δ = 0.26 * s * l() / M
# Δ = 0.5 * s * l() / M
W = sum(gaussian.(Δ, x .- ξ' .- z .* l()) for z ∈ -2:2)
W = W ./ sum(W; dims = 2)
W[abs.(W).<1e-6] .= 0
W = W ./ sum(W; dims = 2)
W = sparse(W)
dropzeros!(W)
plotmat(W; title = "Discrete filter")
plotmat(W .!= 0; title = "Discrete filter (sparsity)")

# # Create data
#
# We will create data sets for compression, training, validation, and testing.

"""
Linear-ish frequency decay.
"""
decay(k) = 1 / (1 + abs(k))^1.2

# Maximum frequency in initial conditions
K = N ÷ 2

# Number of samples
n_compress = 5000
n_train = 120
n_valid = 10
n_test = 60

# Initial conditions (real-valued)
u₀ = create_data(ξ, K, 1; decay)[:]
u₀_compress = create_data(ξ, K, n_compress; decay)
u₀_train = create_data(ξ, K, n_train; decay)
u₀_valid = create_data(ξ, K, n_valid; decay)
u₀_test = create_data(ξ, K, n_test; decay)
plot(ξ, u₀_train[:, 1:3]; xlabel = "x", title = "Initial conditions")

function create_conv(r, s; rng = Random.default_rng())
    if r > s - 1
        d = r - (s - 1)
        pad = u -> [u[end-d+1:end, :, :]; u; u[1:r, :, :]]
    else
        pad = u -> [u[s-r:end, :, :]; u[1:r, :, :]]
    end

    # Compressor
    _NN = Chain(
        # From (nx, nsample) to (nx, nchannel = 1, nsample)
        u -> reshape(u, size(u, 1), 1, size(u, 2)),

        # Manual padding along spatial dimension to account for periodicity
        pad,

        # Linear convolutional layer (no activation or bias)
        Conv(
            (2r + 1,),
            1 => 1,
            identity;
            init_weight = glorot_uniform_Float64,
            bias = false,
            stride = s,
        ),

        # From (nx, nchannel = 1, nsample) to (nx, nsample)
        u -> reshape(u, size(u, 1), size(u, 3)),
    )

    params, state = Lux.setup(rng, _NN)
    p, re = destructure(params)
    NN(u, p) = first(_NN(u, re(p), state))
    NN, p
end

function create_square_conv(r; rng = Random.default_rng())
    # Conv network
    _NN = Chain(
        # From (nx, nsample) to (nx, nchannel = 1, nsample)
        u -> reshape(u, size(u, 1), 1, size(u, 2)),

        # Manual padding along spatial dimension to account for periodicity
        u -> [u[end-r+1:end, :, :]; u; u[1:r, :, :]],

        # Linear convolutional layer (no activation or bias)
        Conv(
            (2r + 1,),
            1 => 1,
            identity;
            init_weight = glorot_uniform_Float64,
            bias = false,
        ),

        # From (nx, nchannel = 1, nsample) to (nx, nsample)
        u -> reshape(u, size(u, 1), size(u, 3)),
    )

    params, state = Lux.setup(rng, _NN)
    p, re = destructure(params)
    NN(u, p) = first(_NN(u, re(p), state))
    NN, p
end

T, τ₀ = create_conv(10, s)

W[25, :].nzval

T(u₀, τ)

plot(ξ, u₀)
plot!(x, ū₀)
plot!(x, T(u₀, zero(τ)))

plot(W[25, :].nzval)

function loss_comp(τ, u)
    ū = W * u
    w = T(u, τ)
    Eū = ū .* ū
    Ew = w .* w
    Eu = u .* u
    sum(abs2, sum(Eū; dims = 1) ./ M + sum(Ew; dims = 1) ./ M - sum(Eu; dims = 1) ./ N) /
    size(u, 2)
end

function loss_comp(τ; u = u₀_compress, nuse = 500)
    nsample = size(u, 2)
    i = Zygote.@ignore sort(shuffle(1:nsample)[1:nuse])
    loss_comp(τ, u[:, i])
end

E(u) = u'u / 2 * l() / length(u)

loss_comp(τ₀; u = u₀_compress[:, 1:100], nuse = 100)
loss_comp(zero(τ); u = u₀_compress[:, 1:100], nuse = 100)
loss_comp(τ; u = u₀_compress[:, 1:100], nuse = 100)
first(gradient(loss_comp, τ))
first(gradient(loss_comp, zero(τ)))

τ = train(
    loss_comp,
    # τ₀,
    τ,
    5000;
    opt = Optimisers.ADAM(0.001),
    callback = (i, τ) -> println(
        "Iteration \t $i, loss: $(loss_comp(τ; u = u₀_compress[:, 1:100], nuse = 100))",
    ),
    ncallback = 10,
)

ū₀ = W * u₀
w₀ = T(u₀, τ)

plot(; xlabel = "x")
plot!(ξ, u₀; label = "u")
plot!(x, ū₀; label = "ū")
plot!(x, w₀; label = "w")

plot(; xlabel = "x")
plot!(ξ, u₀ .^ 2 ./ 2; label = "e(u)")
plot!(x, ū₀ .^ 2 ./ 2; label = "e(ū)")
plot!(x, ū₀ .^ 2 ./ 2 .+ w₀ .^ 2 ./ 2; label = "e(ū) + e(w)")

# Evaluation times
t = LinRange(0.0, 1.0, 101)
t_train = LinRange(0, 0.1, 41)
t_valid = LinRange(0, 0.2, 26)
t_test = LinRange(0, 1.0, 61)

# Unfiltered solutions
u = solve_matrix(FN, u₀, t; reltol = 1e-4, abstol = 1e-6)
u_train = solve_matrix(FN, u₀_train, t_train; reltol = 1e-4, abstol = 1e-6)
u_valid = solve_matrix(FN, u₀_valid, t_valid; reltol = 1e-4, abstol = 1e-6)
u_test = solve_matrix(FN, u₀_test, t_test; reltol = 1e-4, abstol = 1e-6)

# Filtered solutions
ū = apply_matrix(W, u)
ū_train = apply_matrix(W, u_train)
ū_valid = apply_matrix(W, u_valid)
ū_test = apply_matrix(W, u_test)

# Latent solutions
w = reshape(T(Array(u), τ), M, :)
w_train = reshape(T(reshape(Array(u_train), N, :), τ), M, n_train, :)
w_valid = reshape(T(reshape(Array(u_valid), N, :), τ), M, n_valid, :)
w_test = reshape(T(reshape(Array(u_test), N, :), τ), M, n_test, :)

# Augmented states
q = [ū; w]
q_train = [ū_train; w_train]
q_valid = [ū_valid; w_valid]
q_test = [ū_test; w_test]

# Time derivatives
dudt = apply_matrix(FN, u)
dudt_train = apply_matrix(FN, u_train)
dudt_valid = apply_matrix(FN, u_valid)
dudt_test = apply_matrix(FN, u_test)

# Filtered time derivatives
dūdt = apply_matrix(W, dudt)
dūdt_train = apply_matrix(W, dudt_train)
dūdt_valid = apply_matrix(W, dudt_valid)
dūdt_test = apply_matrix(W, dudt_test)

# Filtered latent time derivatives
dwdt = reshape(T(Array(dudt), τ), M, :)
dwdt_train = reshape(T(reshape(Array(dudt_train), N, :), τ), M, n_train, :)
dwdt_valid = reshape(T(reshape(Array(dudt_valid), N, :), τ), M, n_valid, :)
dwdt_test = reshape(T(reshape(Array(dudt_test), N, :), τ), M, n_test, :)

# Augmented time derivatives
dqdt = [dūdt; dwdt]
dqdt_train = [dūdt_train; dwdt_train]
dqdt_valid = [dūdt_valid; dwdt_valid]
dqdt_test = [dūdt_test; dwdt_test]

plot(; xlabel = "t")
plot!(t, E.(eachcol(u)); label = "E(u)")
plot!(t, E.(eachcol(ū)); label = "E(ū)")
plot!(t, E.(eachcol(ū)) .+ E.(eachcol(w)); label = "E(ū) + E(u')")
ylims!((0, ylims()[2]))

# Neural networks for K
# Sice K is square, K^T is given by K(reverse(p))
r11, r12, r22 = (3, 3, 3)
d11, d12, d22 = (2 * r11 + 1, 2 * r12 + 1, 2 * r22 + 1)
K11, p₀11 = create_square_conv(r11)
K12, p₀12 = create_square_conv(r12)
K22, p₀22 = create_square_conv(r22)

# This should be zero
ū₀' * (K11(ū₀, p₀11) - K11(ū₀, reverse(p₀11)))
ū₀' * (K12(ū₀, p₀12) - K12(ū₀, reverse(p₀12)))
ū₀' * (K22(ū₀, p₀22) - K22(ū₀, reverse(p₀22)))

function Q(q, p, t)
    p11 = p[1:d11]
    p12 = p[d11+1:d11+d12]
    p22 = p[d11+d12+1:end]
    u = q[1:M, :]
    w = q[M+1:end, :]
    duu = K11(u, p11) - K11(u, reverse(p11))
    duw = K12(w, p12)
    dwu = -K12(u, reverse(p12))
    dww = K22(w, p22) - K22(w, reverse(p22))
    [duu + duw; dwu + dww]
end

# Callback for studying convergence
function create_callback(Q, q, t; iplot = 1:10)
    ū = q[1:M, iplot, :]
    # w = q[M+1:end, iplot, :]
    hist_i = Int[]
    hist_err = zeros(0)
    err_nomodel =
        relerr(Array(solve_matrix(FM, ū[:, :, 1], t; reltol = 1e-4, abstol = 1e-6)), ū)
    function callback(i, p)
        sol = solve_equation(Q, q[:, iplot, 1], p, t; reltol = 1e-4, abstol = 1e-6)
        v = sol[1:M, :, :]
        err = relerr(v, ū)
        println("Iteration $i \t average relative error $err")
        push!(hist_i, i)
        push!(hist_err, err)
        pl = plot(; title = "Average relative error", xlabel = "Iterations")
        hline!(pl, [err_nomodel]; color = 1, linestyle = :dash, label = "No closure")
        plot!(pl, hist_i, hist_err; label = "With closure")
        display(pl)
    end
end

q₀ = [ū₀; w₀]
p₀ = [p₀11; p₀12; p₀22]

Q(q₀, p₀, 0.0)

prediction_loss(
    Q,
    p₀,
    reshape(dqdt_train, 2M, :),
    reshape(q_train, 2M, :);
    nuse = 100,
    λ = 1e-8,
)

first(
    gradient(
        p -> prediction_loss(
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
    p -> prediction_loss(
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
    # callback = (i, p) -> cb(i + 20_000, p),
    # callback = (i, p) -> println("Iteration $i"),

    # Number of iterations between callbacks
    ncallback = 10,
)

pref = [M / l() * reverse([0, 0, 0, 0, -45, 9, -1]) / 60; zeros(7); zeros(7)]

# p = pref

cb(20_001, p)
cb(20_004, pref)
