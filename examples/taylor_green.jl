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

apply_mat(u, p, t) = p * u
function solve_matrix(A, u₀, t, solver = Tsit5(); kwargs...)
    problem = ODEProblem(apply_mat, u₀, extrema(t), A)
    # problem = ODEProblem(DiffEqArrayOperator(A), u₀, extrema(t))
    solve(problem, solver; saveat = t, kwargs...)
end

# Domain length
l() = 1.0

# Taylor-Green velocity field
nwave() = 1
uu(x, y) = -sinpi(nwave() * x / l()) * cospi(nwave() * y / l())
vv(x, y) = +cospi(nwave() * x / l()) * sinpi(nwave() * y / l())

# Grid sizes
s = 10
M = 20
N = s * M

loc = "output/taylor_green/N$(N)_M$(M)/"
mkpath(loc)

# Grids
xfine = LinRange(0, l(), N + 1)[2:end]
yfine = LinRange(0, l(), N + 1)[2:end]
x = LinRange(0, l(), M + 1)[2:end]
y = LinRange(0, l(), M + 1)[2:end]

plotfield(xfine, yfine, uu.(xfine, yfine'))
plotfield(x, y, uu.(x, y'))

plotfield(xfine, yfine, vv.(xfine, yfine'))
plotfield(x, y, vv.(x, y'))

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

kx = gaussian.(Δx, (-rx:rx) ./ N)
ky = gaussian.(Δy, (-ry:ry) ./ N)

kernel = kx .* ky'
kernel = kernel ./ sum(kernel)
heatmap(
    -rx:rx,
    -ry:ry,
    kernel';
    xlims = (-rx, rx),
    ylims = (-ry, ry),
    title = "Filter kernel",
    xlabel = "x offset",
    ylabel = "y offset",
    aspect_ratio = :equal,
)

# savefig(loc * "taylor_green/kernel.pdf")

i = 1:M
j = reshape(1:M, 1, M)

ax = reshape(-rx:rx, 1, 1, :)
ay = reshape(-ry:ry, 1, 1, 1, :)
ax = @. mod1(s * i + ax, N)
ay = @. mod1(s * j + ay, N)

z = repeat(reshape(kernel, 1, 1, 2rx+1, 2ry+1), M, M, 1, 1)

α = @. i + M * (j - 1)
α = repeat(α, 1, 1, 2rx+1, 2ry+1)
β = @. ax + N * (ay - 1)

W = sparse(α[:], β[:], z[:])

size(W)

plotmat(W[1:50, 1:500]; title = "W")

plotmat(W; title = "W")

# savefig(loc * "taylor_green/W.pdf")

Φ, σ, VV = svd(Matrix(W); full = false)
Σ = Diagonal(σ)
Ψ = VV[:, 1:M*M]
Ξ = VV[:, M*M+1:end]
V = Ξ'
P = Ξ

# ii = 100
# anim = @animate for ii = 1:50
for ii = 1:100
    pl = plotfield(
        xfine,
        yfine,
        reshape(Ψ[:, ii], N, N);
        # aspect_ratio = :equal,
        # xlims = (0, l()),
        title = "ψᵢ, i = $ii",
    )
    display(pl)
    sleep(0.1)
end

gif(anim, loc * "animations/taylor_green_svd.gif"; fps = 10)



# ii = 452
for ii = 1:50
    pl = plotfield(
        xfine,
        yfine,
        reshape(Ξ[:, ii], N, N);
        aspect_ratio = :equal,
        xlims = (0, l()),
        title = "$ii",
    )
    display(pl)
    sleep(0.1)
end

# i = 1:12
i = [1:9; 100; 200; 400]
# i = 13:24
# i = 101:112
# i = [500, 1000, 1500, 2000, 2500, 3000]
mat = Ψ
# mat = Ξ
plot(
    (plotfield(
        xfine, yfine,
        reshape(mat[:, i], N, N);
        # label = i',
        label = false,
        xticks = iplot ≥ 9,
        xlabel = iplot ∈ 9:12 ? "x" : "",
        ylabel = iplot ∈ [1, 5, 9] ? "y" : "",
        yticks = iplot ∈ [1, 5, 9],
        # xlabel = "x",
        title = i,
        aspect_ratio = :equal,
        xlims = (0, l()),
        ylims = (0, l()),
        # colorbar = iplot ∈ [4, 8, 12],
        colorbar = false,
    ) for (iplot, i) = enumerate(i))...,
    clims = extrema(mat[:, i]),
    # layout = (2, 1),
    # title = "Eigenvectors of I/N - W'W/M",
    # size = (1200, 800),
    size = (800, 600),
)

# savefig(loc * "taylor_green/singular_vectors.pdf")
# savefig(loc * "taylor_green/singular_vectors_zero.pdf")

scatter(σ; yscale = :log10, title = "Singular values", legend = false)

# savefig(loc * "taylor_green/singular_values.pdf")

Vxfine = uu.(xfine, yfine')
Vyfine = vv.(xfine, yfine')
Vfine = [Vxfine;;; Vyfine]

Vx = uu.(x, y')
Vy = vv.(x, y')
V = [Vx;;; Vy]

# Check that V is divergence free
heatmap(
    circshift(Vx, -1) - circshift(Vx, 1) + 
    circshift(Vy, (0, -1)) - circshift(Vy, (0, 1))
)

plotfield(xfine, yfine, Vxfine)
plotfield(x, y, Vx)

plotfield(xfine, yfine, Vyfine)
plotfield(x, y, Vy)

plotfield(xfine, yfine, sign.(Vxfine))
plotfield(x, y, sign.(Vx))

function get_F1(Vx, Vy)
    nx, ny = size(Vx)
    i = 1:nx
    j = reshape(1:ny, 1, ny)

    b = reshape(1:nx*ny, nx, ny)
    ax = [circshift(b, -k) for k = -1:1]
    ay = [circshift(b, (0, -k)) for k = -1:1]

    dx1  = (sign.(Vx) .≥ 0) .* ax[1] .+ (sign.(Vx) .< 0) .* ax[2]
    dx2  = (sign.(Vx) .≥ 0) .* ax[2] .+ (sign.(Vx) .< 0) .* ax[3]
    dx = [dx1[:]; dx2[:]]

    dy1  = (sign.(Vy) .≥ 0) .* ay[1] .+ (sign.(Vy) .< 0) .* ay[2]
    dy2  = (sign.(Vy) .≥ 0) .* ay[2] .+ (sign.(Vy) .< 0) .* ay[3]
    dy = [dy1[:]; dy2[:]]

    cx1  = @. -(sign(Vx) ≥ 0) - (sign(Vx) < 0)
    cx2  = @. +(sign(Vx) ≥ 0) + (sign(Vx) < 0)
    cx = nx / l() * [cx1[:]; cx2[:]]

    cy1  = @. -1 * (sign(Vy) ≥ 0) - 1 * (sign(Vy) < 0)
    cy2  = @. +1 * (sign(Vy) ≥ 0) + 1 * (sign(Vy) < 0)
    cy = ny / l() * [cy1[:]; cy2[:]]

    Dx = sparse(dx, [b[:]; b[:]], cx, nx * ny, nx * ny)
    Dy = sparse(dy, [b[:]; b[:]], cy, nx * ny, nx * ny)

    F = -Diagonal(Vx[:]) * Dx - Diagonal(Vy[:]) * Dy
    F
end

function get_F2(Vx, Vy)
    nx, ny = size(Vx)
    i = 1:nx
    j = reshape(1:ny, 1, ny)

    b = reshape(1:nx*ny, nx, ny)
    ax = [circshift(b, -k) for k = -2:2]
    ay = [circshift(b, (0, -k)) for k = -2:2]

    dx1  = (sign.(Vx) .≥ 0) .* ax[1] .+ (sign.(Vx) .< 0) .* ax[3]
    dx2  = (sign.(Vx) .≥ 0) .* ax[2] .+ (sign.(Vx) .< 0) .* ax[4]
    dx3  = (sign.(Vx) .≥ 0) .* ax[3] .+ (sign.(Vx) .< 0) .* ax[5]
    dx = [dx1[:]; dx2[:]; dx3[:]]

    dy1  = (sign.(Vy) .≥ 0) .* ay[1] .+ (sign.(Vy) .< 0) .* ay[3]
    dy2  = (sign.(Vy) .≥ 0) .* ay[2] .+ (sign.(Vy) .< 0) .* ay[4]
    dy3  = (sign.(Vy) .≥ 0) .* ay[3] .+ (sign.(Vy) .< 0) .* ay[5]
    dy = [dy1[:]; dy2[:]; dy3[:]]

    cx1  = @. +1/2 * (sign(Vx) ≥ 0) - 3/2 * (sign(Vx) < 0)
    cx2  = @. -4/2 * (sign(Vx) ≥ 0) + 4/2 * (sign(Vx) < 0)
    cx3  = @. +3/2 * (sign(Vx) ≥ 0) - 1/2 * (sign(Vx) < 0)
    cx = nx / l() * [cx1[:]; cx2[:]; cx3[:]]

    cy1  = @. +1/2 * (sign(Vy) ≥ 0) - 3/2 * (sign(Vy) < 0)
    cy2  = @. -4/2 * (sign(Vy) ≥ 0) + 4/2 * (sign(Vy) < 0)
    cy3  = @. +3/2 * (sign(Vy) ≥ 0) - 1/2 * (sign(Vy) < 0)
    cy = ny / l() * [cy1[:]; cy2[:]; cy3[:]]

    Dx = sparse(dx, [b[:]; b[:]; b[:]], cx, nx * ny, nx * ny)
    Dy = sparse(dy, [b[:]; b[:]; b[:]], cy, nx * ny, nx * ny)

    F = -Diagonal(Vx[:]) * Dx - Diagonal(Vy[:]) * Dy
    F
end

Ffine = get_F2(Vxfine, Vyfine)
F = get_F2(Vx, Vy)

plotmat(Ffine[1:100, 1:100])
plotmat(F[1:20, 1:20])

icfunc(x, y) = exp(-((x - l() / 3)^2 + (y - l() / 4)^2) * 100 / l()^2)
icfunc(x, y) = abs(x - 0.3) < 0.1 && abs(y - 0.4) < 0.1 

nfreq = 10
c = [randn() ./ sqrt(kx^2 + ky^2) for kx = 1:nfreq+1, ky = 1:nfreq+1]
ϕx = 2π * rand(nfreq+1)
ϕy = 2π * rand(nfreq+1)
icfunc(x, y) = sum(c[i,j] * cos(2π * (i - 1) * x - ϕx[i]) * cos(2π * (j - 1) * y - ϕy[j]) for i = 1:nfreq+1, j = 1:nfreq+1)

icfunc(0.2,0.3)

u₀fine = icfunc.(xfine, yfine')
u₀ = icfunc.(x, y')

plotfield(xfine, yfine, u₀fine)
surface(xfine, yfine, u₀fine)

plotfield(x, y, reshape(W * u₀fine[:], M, M))

plotfield(x, y, u₀)
surface(x, y, u₀)

du₀fine = reshape(Ffine * u₀fine[:], N, N)
du₀ = reshape(F * u₀[:], M, M)

plotfield(xfine, yfine, du₀fine)
plotfield(x, y, du₀)

t = LinRange(0, 2.0, 101)
# ufine = solve_equation(f, u₀fine, Vfine, t; reltol = 1e-3, abstol = 1e-6)
# u = solve_equation(f, u₀, V, t; reltol = 1e-3, abstol = 1e-6)
ufine = solve_matrix(Ffine, u₀fine[:], t; reltol = 1e-4, abstol = 1e-8)
u = solve_matrix(F, u₀[:], t; reltol = 1e-3, abstol = 1e-6)

# anim = @animate for (i, t) ∈ enumerate(t)
for (i, t) ∈ enumerate(t)
    pl = plotfield(
        # xfine,
        # yfine,
        # reshape(ufine[i], N, N);
        x,
        y,
        # reshape(W * ufine[i], M, M);
        reshape(u[i], M, M);
        title = @sprintf("Solution, t = %.2f", t),
        xlabel = "x",
        ylabel = "y",
    )
    # quiver!(pl, xfine, yfine, (Vfine[:, :, 1], Vfine[:, :, 2]))
    display(pl)
    sleep(0.005) # Time for plot pane to update
end

gif(anim, loc * "animations/taylor_green_u.gif")

E(u) = 1 / 2 * sum(abs2, u) * l()^2 / prod(size(u))

plot(; title = "Kinetic energy", xlabel = "t")
plot!(t, E.(ufine.u); label = "ufine")
plot!(t, E.(u.u); label = "u")
plot!(t, E.(eachcol(W * Array(ufine))); label = "W ufine")
ylims!((0.0, ylims()[2]))
