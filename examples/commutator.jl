if isdefined(@__MODULE__, :LanguageServer)
    include("../src/ConvolutionalClosure.jl")
    using .ConvolutionalClosure
end

using ConvolutionalClosure
using LinearAlgebra
using SparseArrays
using Plots

"""
    circulant(n, inds, stencil)

Create circulant `SparseMatrixCSC`.
"""
circulant(n, inds, stencil) = spdiagm(
    (i => fill(s, n - abs(i)) for (i, s) ∈ zip(inds, stencil))...,
    (i - sign(i) * n => fill(s, abs(i)) for (i, s) ∈ zip(inds, stencil))...,
)

N = 30
M = 10

DN = circulant(N, [-1, 1], [-N / 2, N / 2])
DM = circulant(M, [-1, 1], [-M / 2, M / 2])
plotmat(DN; title = "DN")
plotmat(DM; title = "DM")

s = [1, 4, 1] / 6
# s = [1, 1, 1] / 3
W = circulant(N, -1:1, s)
W = W[2:3:end, :]
plotmat(W; title = "W")

plotmat(W * DN; title = "W DN")
plotmat(DM * W; title = "DM W")

C = W * DN - DM * W
plotmat(C; title = "C")

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
