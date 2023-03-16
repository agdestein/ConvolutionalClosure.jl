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

# Linear interpolant
R = W' / Matrix(W * W')

Φ, σ, VV = svd(Matrix(W); full = true)
Σ = Diagonal(σ)
Ψ = VV[:, 1:M]
Ξ = VV[:, M+1:end]
V = Ξ'
P = Ξ

"""
Linear-ish frequency decay.
"""
decay(k) = 1 / (1 + abs(k))^1.2

# Maximum frequency in initial conditions
kmax = N ÷ 2
kmax = 100

u = create_data(y, kmax, 1; decay)[:]

ke(u) = u^2 / 2

# Convection
# F = circulant(N, -3:3, N / l() * [1, -9, 45, 0, -45, 9, -1] / 60)
F = -circulant(N, [-1, 1], [-N / 2, N / 2])

# Mori-Zwanzig matrices
A = W * F * R
B = W * F * P
C = V * F * R
D = V * F * P

dec = svd(Matrix(I / N - W'W / M))

scatter(dec.S)
