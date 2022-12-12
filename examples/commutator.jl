if isdefined(@__MODULE__, :LanguageServer)
    include("../src/ConvolutionalClosure.jl")
    using .ConvolutionalClosure
end

using ConvolutionalClosure
using LinearAlgebra
using SparseArrays
using Plots

l() = 1.0

N = 90
M = 30
loc = "output/N$(N)_M$(M)/"
mkpath(loc)

ξ = LinRange(0, l(), N + 1)[2:end]
x = LinRange(0, l(), M + 1)[2:end]

DN = circulant(N, [-1, 1], [-N / 2, N / 2])
DM = circulant(M, [-1, 1], [-M / 2, M / 2])
plotmat(DN; title = "DN")
plotmat(DM; title = "DM")

# s = [1, 4, 1] / 6
s = [1, 1, 1] / 3
W = circulant(N, -1:1, s)
W = W[2:3:end, :]
plotmat(W; title = "W")

plotmat(W * DN; title = "W DN")
plotmat(DM * W; title = "DM W")

C = W * DN - DM * W
plotmat(C; title = "C")

P = interpolation_matrix(l(), x, ξ)
plotmat(P)

plotmat(W * DN * P)
plotmat(DM)

pl = plot()
bar!((W * DN)[6, :]; label = "W D")
bar!((DM * W)[6, :]; label = "D W")
pl

bar(C[6, :])

plotmat(W * DN * W')
plotmat(DM)
A = W * DN * W' / (W * W')
plotmat(A)
plotmat(A * 2 * N / M)
plotmat(DM)
plotmat(W * W')

sum(P; dims = 2)
sum(W * DN; dims = 2)
sum(W * DN; dims = 2)
sum(DM * W; dims = 2)
sum(DM * W; dims = 2)

plot(
    plotmat(DN; title = "DN"),
    plotmat(DM; title = "DM"),
    plotmat(W; title = "W"),
    plotmat(W * DN; title = "W DN"),
    plotmat(DM * W; title = "DM W"),
    plotmat(C; title = "C");
    size = (1200, 600),
)
savefig(loc * "mat.png")

##
K = N ÷ 2
u = create_data(ξ, K, 1; decay = k -> 1 / (1 + abs(k))^1.2)
pl = plot()
plot!(ξ, u; label = "u")
plot!(ξ, P * W * u; label = "Pū")
scatter!(x, W * u; label = "ū")
pl
