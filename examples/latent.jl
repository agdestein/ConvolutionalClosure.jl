# Little LSP hack to get function signatures, go    #src
# to definition etc.                                #src
if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/ConvolutionalClosure.jl")       #src
    using .ConvolutionalClosure                     #src
end                                                 #src

# # Mori-Zwanzig expansion
#
# Solve for the filtered variable by expanding the Mori-Zwanzig memory term.

using ConvolutionalClosure
using Expokit
using LinearAlgebra
using OrdinaryDiffEq
using Plots
using Printf
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

s = 4
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
W = sum(top_hat.(Δ, x .- ξ' .- z .* l()) for z ∈ -2:2)
W = W ./ sum(W; dims = 2)
W[abs.(W).<1e-6] .= 0
W = W ./ sum(W; dims = 2)
W = sparse(W)
dropzeros!(W)
plotmat(W; title = "Discrete filter")
plotmat(W .!= 0; title = "Discrete filter (sparsity)")

sum(R; dims = 2)

# # Piece-wise constant interpolant
# R = constant_interpolator(l(), x, ξ)

# Linear interpolant
R = linear_interpolator(l(), x, ξ)

# # Full reconstructor
# R = Matrix(W') / Matrix(W * W')

# Plot
plotmat(R)
plotmat(R * W)
plotmat(W * R)

# First order optimal predictor
A = W * FN * R
# A = FM

# Sub-filter stressor
X = W * FN - A * W

plotmat(A)
plotmat(X)

"""
Linear-ish frequency decay.
"""
decay(k) = 1 / (1 + abs(k))^1.2

# Initial conditions
u₀ = @. sin(2π * ξ / l())
# u₀ = create_data(ξ, N ÷ 2, 1; decay)[:]
# u₀ = create_data(ξ, 50, 1; decay)[:]
# u₀ = R * W * u₀
# for i = 1:5
#     u₀ = R * W * u₀
# end

pl_u = plot(; xlabel = "x", title = "Initial conditions")
plot!(ξ, u₀; label = "Unfiltered")
plot!(x, W * u₀; label = "Filtered")
pl_u

pl_w = plot(x, X * u₀; xlabel = "x", label = "Sub-filter stresses")

plot(pl_u, pl_w; layout = (2, 1))

# Plot example solution
t = LinRange(0, 1.0, 101)
u = solve_matrix(FN, u₀, t, Tsit5(); reltol = 1e-8, abstol = 1e-10)
ū = W * u
v = solve_matrix(A, W * u₀, t, Tsit5(); reltol = 1e-8, abstol = 1e-10)
# v = solve_matrix(FM, W * u₀, t, Tsit5(); reltol = 1e-8, abstol = 1e-10)
for (i, t) ∈ enumerate(t)
    pl = plot(; xlabel = "x", title = @sprintf("t = %.2f", t), ylims = extrema(u[:, :]))
    plot!(pl, ξ, u[i]; label = "Unfiltered")
    plot!(pl, x, ū[:, i]; label = "Filtered")
    plot!(pl, x, v[:, i]; label = "A")
    display(pl)
    sleep(0.05) # Time for plot pane to update
end

BB = Matrix(X * FN * [W; X]') / Matrix([W; X] * [W; X]')
Q = [A I; BB]

plotmat(BB)
plotmat(Q)
plotmat(log.(abs.(Q)))

# Example
t = LinRange(0, 1.01, 101)
u = solve_matrix(FN, u₀, t; reltol = 1e-8, abstol = 1e-10)
ū = W * u
w = X * u
dūdt = W * FN * u
dwdt = X * FN * u

# Baseline
v0 = solve_matrix(A, ū[:, 1], t; reltol = 1e-8, abstol = 1e-10)

# Augmented
q = solve_matrix(Q, [W; X] * u₀, t; reltol = 1e-8, abstol = 1e-10)
v1 = q[1:M, :]

# Plot example solution
for (i, t) ∈ enumerate(t)
    pl = plot(; xlabel = "x", title = @sprintf("t = %.2f", t), ylims = extrema(u[:, :]))
    # plot!(pl, ξ, u[i]; label = "Unfiltered")
    plot!(pl, x, ū[:, i]; label = "Filtered")
    plot!(pl, x, v0[:, i]; label = "v0")
    plot!(pl, x, v1[:, i]; label = "v1")
    display(pl)
    sleep(0.005) # Time for plot pane to update
end

# Plot example solution
for (i, t) ∈ enumerate(t)
    pl = plot(; xlabel = "x", title = @sprintf("t = %.2f", t), ylims = extrema(w[:, :]))
    # plot!(pl, ξ, u[i]; label = "Unfiltered")
    plot!(pl, x, w[:, i]; label = "w")
    display(pl)
    sleep(0.005) # Time for plot pane to update
end
