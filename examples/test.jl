using LinearAlgebra
using Plots
using LaTeXStrings

"""
    central_diff(u, s)

Differentiate `u` with central difference stencil `s`.
"""
function central_diff(u, s)
    n = length(s) ÷ 2
    sum(circshift(u, -i) * s for (i, s) ∈ zip(-n:n, s))
end

# Spatial discretization
N = 100
x = LinRange(0, 1, N + 1)[2:end]
Δx = 1 / N

# Finite difference stencils
s₂ = [-1, 0, 1] / 2Δx
s₄ = [1, -8, 0, 8, -1] / 12Δx # 4th order
s₆ = [-1, 9, -45, 0, 45, -9, 1] / 60Δx # 6th order
s₈ = [3, -32, 168, -672, 0, 672, -168, 32, -3] / 840Δx # 8th order

orders = [2, 4, 6, 8]
ss = [s₂, s₄, s₆, s₈]

# Number of frequencies
K = N ÷ 2
kk = 0:K-1

# Fourier basis functions
e = [exp(2π * im * k * x) / sqrt(N) for x ∈ x, k ∈ kk]
k = 1:3
plot(
    x,
    real.(e[:, k]);
    xlabel = "x",
    title = "eₖ(x)",
    label = map(k -> "k = $(k - 1)", reshape(k, 1, :)),
)

# Check that they are normalized
@assert all(≈(1), norm.(eachcol(e)))

# Exact derivatives of Fourier basis functions
de = [2π * im * k * exp(2π * im * k * x) / sqrt(N) for x ∈ x, k ∈ 0:K-1]

# Approximate derivatives of Fourier basis functions
de_diff = [central_diff(e, s) for s ∈ ss]

# Compare
k = 3
pl = plot(; xlabel = "x", title = "deₖ/dx, k = $(k - 1)")
plot!(pl, x, real.(de[:, k]); label = "Exact")
for (o, de) ∈ zip(orders, de_diff)
    plot!(pl, x, real.(de[:, k]); label = "Order $o")
end
pl

# Compare errors
pl = plot(;
    yscale = :log10,
    xlabel = "Central difference order",
    xticks = orders,
    title = "Relative error of deₖ/dx",
)
for k ∈ [3, 10, 20]
    errors = map(x -> norm(x[:, k] - de[:, k]) / norm(de[:, k]), de_diff)
    plot!(pl, orders, errors; marker = :o, label = "k = $(k - 1)")
end
pl

# Plot frequency attenuation
e_de = [abs(e[:, k]' * de[:, k]) for k = 1:K]
e_de_diff = [[abs(e[:, k]' * de[:, k]) for k = 1:K] for de ∈ de_diff]
pl = plot(;
    xlabel = L"k",
    title = L"\int_0^1 \bar{e}_k(x) \frac{\mathrm{d} e_k}{\mathrm{d} x}(x) \, \mathrm{d} x",
    legend = :topleft,
)
plot!(pl, kk, e_de; label = "Exact")
for (o, e_de) ∈ zip(orders, e_de_diff)
    plot!(pl, kk, e_de; label = "Order $o")
end
pl
