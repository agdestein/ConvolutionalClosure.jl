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

# Domain length
l() = 1.0

# Taylor-Green velocity field
nwave() = 1
uu(x, y) = -sinpi(nwave() * x) * cospi(nwave() * y)
vv(x, y) = cospi(nwave() * x) * sinpi(nwave() * y)

# Grid sizes
s = 2
M = 50
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

"""
Right hand side.
"""
function f(u, p, t)
    nx, ny = size(u)
    ## s = [1, 0, -1] / 2
    s = [1, -9, 45, 0, -45, 9, -1] / 60
    r = length(s) ÷ 2
    dudx = sum(i -> nx / l() * s[r+1+i] * circshift(u, i), -r:r)
    dudy = sum(j -> ny / l() * s[r+1+j] * circshift(u, (0, j)), -r:r)
    @. -p[:, :, 1] * dudx - p[:, :, 2] * dudy
end

u₀ = [exp(-((x - l() / 3)^2 + (y - l() / 4)^2) * 100 / l()^2) for x ∈ xfine, y ∈ yfine]

heatmap(xfine, yfine, u₀)
surface(xfine, yfine, u₀)

du₀ = f(u₀, Vfine, 0.0)
heatmap(xfine, yfine, du₀)
surface(xfine, yfine, du₀)

t = LinRange(0, 10, 201)
u = solve_equation(f, u₀, Vfine, t; reltol = 1e-3, abstol = 1e-6)

for (i, t) ∈ enumerate(t)
    pl = heatmap(
        xfine,
        yfine,
        u[i];
        xlabel = "x",
        ylabel = "y",
        title = @sprintf("Solution, t = %.2f", t),
    )
    ## quiver!(pl, xfine, yfine, (Vfine[:, :, 1], Vfine[:, :, 2]))
    display(pl)
    sleep(0.05) # Time for plot pane to update
end

E(u) = sum(abs2, u) * l()^2 / prod(size(u))

plot(t, E.(u.u); legend = false, title = "Kinetic energy")
ylims!((0.0, ylims()[2]))
