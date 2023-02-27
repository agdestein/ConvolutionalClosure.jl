# Little LSP hack to get function signatures, go    #src
# to definition etc.                                #src
if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/ConvolutionalClosure.jl")       #src
    using .ConvolutionalClosure                     #src
end                                                 #src

# # Taylor-Green convection
#
# Convect a quantity ``u`` using the Taylor-Green velocity field

using ConvolutionalClosure
using LinearAlgebra
using OrdinaryDiffEq
using Plots
using Printf
using SparseArrays

# Domain length
l() = 1.0

# Taylor-Green velocity field
nwave() = 2
uu(x, y) = -sinpi(nwave() * x) * cospi(nwave() * y)
vv(x, y) = cospi(nwave() * x) * sinpi(nwave() * y)

# Grid sizes
s = 5
M = 20
N = s * M

loc = "output/taylor_green/N$(N)_M$(M)/"
mkpath(loc)

# Grids
xfine = LinRange(0, l(), N + 1)[2:end]
yfine = LinRange(0, l(), N + 1)[2:end]
x = LinRange(0, l(), M + 1)[2:end]
y = LinRange(0, l(), M + 1)[2:end]

heatmap(xfine, yfine, uu; xlabel = "x", ylabel = "y")
heatmap(x, y, uu; xlabel = "x", ylabel = "y")

heatmap(xfine, yfine, vv; xlabel = "x", ylabel = "y")
heatmap(x, y, vv; xlabel = "x", ylabel = "y")

Vfine = [uu.(xfine, yfine');;; vv.(xfine, yfine')]
V = [uu.(x, y');;; vv.(x, y')]

# Check that V is divergence free
heatmap(
    circshift(V[:, :, 1], -1) - circshift(V[:, :, 1], 1) + 
    circshift(V[:, :, 2], (0, -1)) - circshift(V[:, :, 2], (0, 1))
)

"""
Right hand side.
"""
function f(u, p, t)
    nx, ny = size(u)
    s = [1, 0, -1] / 2
    # s = [1, -9, 45, 0, -45, 9, -1] / 60
    r = length(s) ÷ 2
    dudx = sum(i -> nx / l() * s[r+1+i] * circshift(u, i), -r:r)
    dudy = sum(j -> ny / l() * s[r+1+j] * circshift(u, (0, j)), -r:r)
    @. -p[:, :, 1] * dudx - p[:, :, 2] * dudy
end

u₀fine = [exp(-((x - l() / 3)^2 + (y - l() / 4)^2) * 100 / l()^2) for x ∈ xfine, y ∈ yfine]
u₀ = [exp(-((x - l() / 3)^2 + (y - l() / 4)^2) * 100 / l()^2) for x ∈ x, y ∈ y]

heatmap(xfine, yfine, u₀fine)
surface(xfine, yfine, u₀fine)

heatmap(x, y, u₀)
surface(x, y, u₀)

du₀fine = f(u₀fine, Vfine, 0.0)
du₀ = f(u₀, V, 0.0)
heatmap(xfine, yfine, du₀fine)
heatmap(x, y, du₀)

t = LinRange(0, 2, 101)
ufine = solve_equation(f, u₀fine, Vfine, t; reltol = 1e-3, abstol = 1e-6)
u = solve_equation(f, u₀, V, t; reltol = 1e-3, abstol = 1e-6)

for (i, t) ∈ enumerate(t)
    pl = surface(
        # xfine,
        # yfine,
        # ufine[i];
        x,
        y,
        u[i];
        xlabel = "x",
        ylabel = "y",
        title = @sprintf("Solution, t = %.2f", t),
    )
    ## quiver!(pl, xfine, yfine, (Vfine[:, :, 1], Vfine[:, :, 2]))
    display(pl)
    sleep(0.005) # Time for plot pane to update
end

E(u) = 1 / 2 * sum(abs2, u) * l()^2 / prod(size(u))

plot(t, E.(ufine.u); legend = false, title = "Kinetic energy")
ylims!((0.0, ylims()[2]))

# Discrete filter matrix
# This assumes uniform grid and `N = s * M`
Δ = 1 / 10
Δx = 2Δ
Δy = Δ
dx = 1 / N
dy = 1 / N

rx = round(Int, N * Δx)
ry = round(Int, N * Δy)

plot(gaussian.(Δx, (-rx:rx) ./ N))
plot(gaussian.(Δy, (-ry:ry) ./ N))
M^2 * length(-rx:rx) * length(-ry:ry)

sparse()

W = W ./ sum(W; dims = 2)
W = sparse(W)
dropzeros!(W)
# W = W / sqrt(λmax) 1.0000000003
plotmat(W; title = "W")
