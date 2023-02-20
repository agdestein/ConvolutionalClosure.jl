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

s = 10
M = 50
N = s * M

loc = "output/expansion/N$(N)_M$(M)/"
mkpath(loc)

ξ = LinRange(0, l(), N + 1)[2:end]
x = LinRange(0, l(), M + 1)[2:end]

# FN = circulant(N, [-1, 1], -[-N / 2, N / 2])
# FM = circulant(M, [-1, 1], -[-M / 2, M / 2])
FN = circulant(N, -3:3, N * [1, -9, 45, 0, -45, 9, -1] / 60)
FM = circulant(M, -3:3, M * [1, -9, 45, 0, -45, 9, -1] / 60)
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

# # Piece-wise constant interpolant
# R = constant_interpolator(l(), x, ξ)

# # Linear interpolant
# R = linear_interpolator(l(), x, ξ)

# Full reconstructor
R = Matrix(W') / Matrix(W * W')

diag(R'R, 1)

R[20, :]

# Plot
plotmat(R)
plotmat(R * W)
plotmat(W * R)

sum(R[1, :])

# Mori-Zwanzig matrices
A = W * FN * R
B = W * FN
C = (I - R * W) * FN * R
D = (I - R * W) * FN

plotmat(A)
plotmat(B)
plotmat(C)
plotmat(D)

plotmat(B * C)
plotmat(-B * D * B' / Matrix(B * B'); clims = extrema(A))
plotmat(A)

plotmat(Matrix(B * B'))
plotmat(inv(Matrix(B * B')))

X = [A I; B * C B * D * B' / Matrix(B * B')]
plotmat(log.(abs.(X)))

"""
Linear-ish frequency decay.
"""
decay(k) = 1 / (1 + abs(k))^1.2

# Initial conditions
u₀ = create_data(ξ, N ÷ 2, 1; decay)[:]
# u₀ = create_data(ξ, 50, 1; decay)[:]
# u₀ = R * W * u₀
# for i = 1:5
#     u₀ = R * W * u₀
# end

pl = plot(; xlabel = "x", title = "Initial conditions")
plot!(ξ, u₀; label = "Unfiltered")
scatter!(x, W * u₀; label = "Filtered")
plot!(ξ, R * W * u₀; label = "Reconstructed")
plot!(ξ, (I - R * W) * u₀; label = "Sub-filter")
pl

# Plot example solution
t = LinRange(0, 1.0, 101)
u = solve_matrix(FN, u₀, t, Tsit5(); reltol = 1e-8, abstol = 1e-10)
ū = W * u
v = solve_matrix(A, W * u₀, t, Tsit5(); reltol = 1e-8, abstol = 1e-10)
for (i, t) ∈ enumerate(t)
    pl = plot(; xlabel = "x", title = @sprintf("t = %.2f", t), ylims = extrema(u[:, :]))
    plot!(pl, ξ, u[i]; label = "Unfiltered")
    plot!(pl, x, ū[:, i]; label = "Filtered")
    plot!(pl, x, v[:, i]; label = "A")
    display(pl)
    sleep(0.05) # Time for plot pane to update
end

# Evaluation time
t = LinRange(0, 0.5, 2001)

# Unfiltered solution
u = solve_matrix(FN, u₀, t; reltol = 1e-8, abstol = 1e-10)

# Filtered solution
ū = W * u

# Unresolved solution
e = (I - R * W) * u

# Filtered time derivatives
dūdt = W * FN * u

# Kinetic energy
E(u) = 1 / 2 * l() / length(u) * u'u
pl = plot(; xlabel = "t", title = "Kinetic energy")
plot!(t, map(E, eachcol(u)); label = "E(u)")
plot!(t, map(E, eachcol(R * ū)); label = "E(ū)")
plot!(t, E.(eachcol(R * ū)) + E.(eachcol(e)); label = "E(ū) + E(u')")
# ylims!((0, ylims()[2]))
pl

# Individual terms in dūdt
markov = A * ū
noise = mapreduce(t -> B * expmv(t, D, e[:, 1]), hcat, t)
memory = dūdt - markov - noise

maximum(abs, noise)

pl_u = plotsol(ξ, t, u; title = "Solution")
pl_ū = plotsol(ξ, t, R * ū; title = "Filtered solution")
pl_markov = plotsol(ξ, t, R * markov; title = "Markovian term")
pl_noise = plotsol(ξ, t, R * noise; title = "Noise term")
pl_memory = plotsol(ξ, t, R * memory; title = "Memory term")

pl_ū = plotsol(x, t, ū; title = "Filtered solution")
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

q₀ = [ū[:, 1]; zeros(nw * M)]
q = solve_matrix(Q, q₀, t, LinearExponential(); reltol = 1e-8, abstol = 1e-10)

v1 = q[1:M, :]

v0 = solve_matrix(A, ū[:, 1], t; reltol = 1e-8, abstol = 1e-10)

# relerr(Array(v0), ū)
# relerr(v1, ū)

relerr(Array(v0)[:, end], ū[:, end])
relerr(v1[:, end], ū[:, end])

pl = plot(; xlabel = "x", title = "Prediction")
plot!(x, ū[:, end]; label = "Truth")
plot!(x, v0[:, end]; label = "0")
plot!(x, v1[:, end]; label = "1")
pl

pl = plot(; xlabel = "x", title = "Prediction")
plot!(ξ, R * ū[:, end]; label = "Truth")
plot!(ξ, R * v0[:, end]; label = "0")
plot!(ξ, R * v1[:, end]; label = "1")
pl
