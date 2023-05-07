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

"Domain length"
l() = 1.0

# Coarse discretization
M = 50

# Fine discretization
s = 10
N = s * M

# Grid
x = LinRange(0, l(), M + 1)[2:end]
y = LinRange(0, l(), N + 1)[2:end]

# F = circulant(N, [-1, 1], -[-N / 2, N / 2])
F = circulant(N, -3:3, N * [1, -9, 45, 0, -45, 9, -1] / 60)
plotmat(F; title = "F")

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

# Non-uniform modal basis
Φ, σ, Ψ = svd(Matrix(W))
Σ = Diagonal(σ)

# Grid refiner
R = √s * Ψ * Φ'
plotmat(R; title = "R")

# Mori-Zwanzig matrices
A = W * F * R
B = W * F
C = (I - R * W) * F * R
D = (I - R * W) * F

plotmat(A)
plotmat(B)
plotmat(C)
plotmat(D)

X = [A I; B*C B*D*B'/Matrix(B * B')]
plotmat(X)
plotmat(log.(abs.(X)))

"""
Linear-ish frequency decay.
"""
decay(k) = 1 / (1 + abs(k))^1.2

# Initial conditions
u₀ = create_data(y, N ÷ 2, 1; decay)[:]

pl = plot(; xlabel = "x", title = "Initial conditions")
plot!(y, u₀; label = "Unfiltered")
scatter!(x, W * u₀; label = "Filtered")
plot!(y, R * W * u₀; label = "Reconstructed")
plot!(y, (I - R * W) * u₀; label = "Sub-filter")
pl

# Plot example solution
t = LinRange(0, 1.0, 101)
u = solve_matrix(F, u₀, t, Tsit5(); reltol = 1e-8, abstol = 1e-10)
v = W * u
vA = solve_matrix(A, W * u₀, t, Tsit5(); reltol = 1e-8, abstol = 1e-10)
for (i, t) ∈ enumerate(t)
    pl = plot(; xlabel = "x", title = @sprintf("t = %.2f", t), ylims = extrema(u[:, :]))
    plot!(pl, y, u[i]; label = "Unfiltered")
    plot!(pl, x, v[:, i]; label = "Filtered")
    plot!(pl, x, vA[:, i]; label = "A")
    display(pl)
    sleep(0.05) # Time for plot pane to update
end

# Evaluation time
t = LinRange(0, 0.5, 2001)

# Unfiltered solution
u = solve_matrix(F, u₀, t; reltol = 1e-8, abstol = 1e-10)

# Filtered solution
v = W * u

# Unresolved solution
e = (I - R * W) * u

# Filtered time derivatives
dvdt = W * F * u

# Kinetic energy
E(u) = 1 / 2 * l() / length(u) * u'u
pl = plot(; xlabel = "t", title = "Kinetic energy")
plot!(t, map(E, eachcol(u)); label = "E(u)")
plot!(t, map(E, eachcol(R * v)); label = "E(ū)")
plot!(t, E.(eachcol(R * v)) + E.(eachcol(e)); label = "E(ū) + E(u')")
# ylims!((0, ylims()[2]))
pl

# Individual terms in dvdt
markov = A * v
noise = mapreduce(t -> B * expmv(t, D, e[:, 1]), hcat, t)
memory = dvdt - markov - noise

maximum(abs, noise)

pl_u = plotsol(y, t, u; title = "Solution")
pl_v = plotsol(y, t, R * v; title = "Filtered solution")
pl_markov = plotsol(y, t, R * markov; title = "Markovian term")
pl_noise = plotsol(y, t, R * noise; title = "Noise term")
pl_memory = plotsol(y, t, R * memory; title = "Memory term")

pl_v = plotsol(x, t, v; title = "Filtered solution")
pl_markov = plotsol(x, t, markov; title = "Markovian term")
pl_noise = plotsol(x, t, noise; title = "Noise term")
pl_memory = plotsol(x, t, memory; title = "Memory term")

pl = plot(; xlabel = "t", title = "Term size (dū/dt)")
plot!(t, norm.(eachcol(markov)); label = "Markov")
plot!(t, norm.(eachcol(noise)); label = "Noise")
plot!(t, norm.(eachcol(memory)); label = "Memory")
pl

nw = 1
AA = vcat(A, (B * D^(k - 1) * C for k = 1:nw)...)
BB = [I(nw * M); zeros(M, nw * M)]
Q = [AA BB]

plotmat(AA)
plotmat(BB)
plotmat(Q)
plotmat(log.(abs.(Q)))

q₀ = [v[:, 1]; zeros(nw * M)]
q = solve_matrix(Q, q₀, t, Tsit5(); reltol = 1e-8, abstol = 1e-10)

v1 = q[1:M, :]

v0 = solve_matrix(A, v[:, 1], t; reltol = 1e-8, abstol = 1e-10)

# relerr(Array(v0), ū)
# relerr(v1, ū)

relerr(Array(v0)[:, end], v[:, end])
relerr(v1[:, end], v[:, end])

pl = plot(; xlabel = "x", title = "Prediction")
plot!(x, v[:, end]; label = "Truth")
plot!(x, v0[:, end]; label = "0")
plot!(x, v1[:, end]; label = "1")
pl

pl = plot(; xlabel = "x", title = "Prediction")
plot!(y, R * v[:, end]; label = "Truth")
plot!(y, R * v0[:, end]; label = "0")
plot!(y, R * v1[:, end]; label = "1")
pl
