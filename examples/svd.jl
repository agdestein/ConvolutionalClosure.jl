# Little LSP hack to get function signatures, go    #src
# to definition etc.                                #src
if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/ConvolutionalClosure.jl")       #src
    using .ConvolutionalClosure                     #src
end                                                 #src

using ConvolutionalClosure
using JLD2
using LinearAlgebra
using Plots
using Printf
using Random
using SparseArrays

Random.seed!(123)

l() = 1.0

M = 50
s = 10
N = s * M

x = LinRange(0, l(), M + 1)[2:end]
y = LinRange(0, l(), N + 1)[2:end]

# Filter widths
ΔΔ(x) = 5 / 100 * l() * (1 + 1 / 3 * sin(2π * x / l()))
Δ = ΔΔ.(x)
plot(x, Δ; legend = false, xlabel = "x", title = "Filter width Δ(x)")

# Δ = 3 / 100 * l()
# Δ = 0.5 * s * l() / M

# W = kron(I(M), ones(1, s)) / s
# plotmat(W)

# Discrete filter matrix
# The support is truncated to 2Δ
W = sum(-1:1) do z
    d = x .- y' .- z .* l()
    gaussian.(Δ, d) .* (abs.(d) .≤ 3 ./ 2 .* Δ)
    # top_hat.(Δ, d)
end
W = W ./ sum(W; dims = 2)
W = sparse(W)
dropzeros!(W)
# W = W / sqrt(λmax) 1.0000000003
plotmat(W; title = "W")

plotmat(W .!= 0; title = "Discrete filter (sparsity)")

plot(W[10, :]; yscale = :log10, ylims = (1e-8, 1))
plot(W[10, :])

# Linear interpolant
R = W' / Matrix(W * W')
plotmat(R)

plotmat(W * W')
plotmat(W' * W)
plotmat(R * W)
plotmat(W * R)

Φ, σ, VV = svd(Matrix(W); full = true)
Σ = Diagonal(σ)
Ψ = VV[:, 1:M]
Ξ = VV[:, M+1:end]
V = Ξ'
P = Ξ

k = 9
for k = 1:M
    ind = 1:k
    # ind = k:k
    Wk = Φ[:, ind] * Diagonal(σ[ind]) * Ψ[:, ind]'
    display(plotmat(Wk; title = "k = $k"))
    sleep(0.1)
end

i = [1:7; 10; 20; 30; 40; 50]
indind = k -> 1:k
# indind = k -> k:k
plot(
    (plotmat(
        Φ[:, indind(k)] * Diagonal(σ[indind(k)]) * Ψ[:, indind(k)]';
        xticks = iplot ≤ 4,
        yticks = iplot ∈ [1, 5, 9],
        # xlabel = "x",
        colorbar = false,
        title = k,
    ) for (iplot, k) = enumerate(i))...,
    # layout = (2, 1),
    # title = "Eigenvectors of I/N - W'W/M",
    # size = (1200, 800),
    size = (800, 500),
    # plot_title = "Truncated filter matrix",
)

# savefig(loc * "nonuniform/Wk.pdf")
# savefig(loc * "nonuniform/Wkall.pdf")

plotmat(V)
plotmat(P)
plotmat(P * V)
plotmat(Ψ)

i =  20; plot(y, Ψ[:, i]; title = "Right singular vectors $i", xlabel = "x")
i = 316; plot(y, P[:, i]; title = "Right singular vectors $i", xlabel = "x")

for i = 1:M
    display(plot(y, Ψ[:, i]; title = "Right singular vectors $i", xlabel = "x"))
    sleep(0.1)
end

for k = 300:400
    display(plot(y, P[:, k]; legend = false, title = "ξₖ, k = $k", xlabel = "x"))
    sleep(0.05)
end


i = [1:7; 10; 20; 30; 40; 50]
plot(
    (plot(
        y,
        Ψ[:, i];
        # label = i',
        label = false,
        xticks = iplot ≥ 9,
        xlabel = iplot ≥ 9 ? "x" : "",
        yticks = false,
        # xlabel = "x",
        title = i,
    ) for (iplot, i) = enumerate(i))...,
    # layout = (2, 1),
    # title = "Eigenvectors of I/N - W'W/M",
    # size = (1200, 800),
    size = (800, 500),
)

# savefig(loc * "nonuniform/singular_vectors.pdf")

i = [1, 50, 100, 150, 200, 250]
plot(
    (plot(
        y,
        Ξ[:, i];
        label = false,
        xticks = iplot ≥ 4,
        xlabel = iplot ≥ 4 ? "x" : "",
        yticks = false,
        # xlabel = "x",
        title = "k = $i",
    ) for (iplot, i) = enumerate(i))...,
    # layout = (2, 1),
    # title = "Eigenvectors of I/N - W'W/M",
    # size = (1200, 800),
    plot_title = "Right singular vectors ξₖ",
    size = (800, 500),
)

# savefig(loc * "nonuniform/singular_vectors_zero.pdf")

scatter(σ; title = "Singular values", legend = false)

# savefig(loc * "nonuniform/singular_values.pdf")

"""
Linear-ish frequency decay.
"""
decay(k) = 1 / (1 + abs(k))^1.2

# Maximum frequency in initial conditions
kmax = N ÷ 2

u = create_data(y, kmax, 1; decay)[:]

k = 3
for k = 1:M
    i = 1:k
    pl = plot(; xlabel = "x", title = "Signal")
    plot!(y, u; label = "u")
    # plot!(y, R * W * u;  label = "RWu")
    plot!(y, Ψ[:, i] * Ψ[:, i]' * u;  label = "ΨiΨi'u")
    plot!(y, P * V * u;  label = "PVu")
    display(pl)
    sleep(0.1)
end

ke(u) = u^2 / 2

plot(; xlabel = "x", title = "Kinetic energy")
plot!(y, ke.(u); label = "u")
plot!(y, ke.(R * W * u);  label = "RWu")
plot!(y, ke.(P * V * u);  label = "PVu")
# plot!(x, ke.(W * u);  label = "Wu")

sum(ke.(W * u)) / M
sum(ke.(R * W * u)) / N
sum(ke.(P * V * u)) / N
sum(ke.(R * W * u)) / N + sum(ke.(P * V * u)) / N
sum(ke, u) / N

plot(; xlabel = "SVD index", yscale = :log10)
scatter!(σ; label = "σ")
scatter!(abs.(Σ * Ψ' * u); label = "|ΣΨ'u|")
scatter!(abs.(Ψ' * u); label = "|Ψ'u|")

σ

e = zeros(N)
e[100] = 1

plot(; xlabel = "x", title = "Signal")
plot!(y, e; label = "e")
# plot!(x, W * e;  label = "RWe")
plot!(y, R * W * e;  label = "RWe")
# plot!(y, e - R * W * e; label = "RWe")


plot(x, abs.(R[500, :]); yscale = :log10)
plot(y, abs.(R[:, 31]); yscale = :log10)

# Convection
# F = circulant(N, -3:3, N / l() * [1, -9, 45, 0, -45, 9, -1] / 60)
F = -circulant(N, [-1, 1], [-N / 2, N / 2])

# Mori-Zwanzig matrices
A = W * F * R
B = W * F * P
C = V * F * R
D = V * F * P

plA = plotmat(A; title = "A")
plB = plotmat(B; title = "B")
plC = plotmat(C; title = "C")
plD = plotmat(D; title = "D")

plot(plA, plB, plC, plD; size = (800, 500))

# savefig(loc * "nonuniform/mzmat.pdf")

sticks(x, A[10, :]; xlims = (0, 1))

sticks(y, W[10, :]; xlims = (0, 1))
sticks(x, W[:, 100]; xlims = (0, 1))

m = 40
sticks(x, R[s * m, :]; xlims = (0, 1))
plot(y, R[:, m]; xlims = (0, 1))

for m = 1:M
    display(plot(y, R[:, m]; xlims = (0, 1), ylims = extrema(R)))
    sleep(0.005)
end

for n = 100:200
    display(sticks(x, R[n, :]; xlims = (0, 1), ylims = extrema(R)))
    sleep(0.005)
end

surface(R; xflip = true)
surface(W; xflip = true)
surface(W)
plotmat(W)
