"""
    Diffusion(l, μ, N = 2)

Diffusion equation with domain length `l`, diffusivity `μ`, and order `N`.
"""
struct Diffusion{N,T} <: AbstractEquation
    l::T
    μ::T
    Diffusion(l, N = 2) = new{N,typeof(l)}(l, μ)
end

"""
    (::Diffusion)(u, p, t)

Compute right hand side of diffusion equation.
This works for both vector and matrix `u` (one or many solutions).
"""
function (::Diffusion) end

function (e::Diffusion{2})(u, p, t)
    Δx = e.l / size(u, 1)
    u₋₁ = circshift(u, 1)
    u₊₁ = circshift(u, -1)
    @. (u₊₁ - 2u + u₋₁) / Δx^2
end

function (e::Diffusion{4})(u, p, t)
    Δx = e.l / size(u, 1)
    u₋₂ = circshift(u, 2)
    u₋₁ = circshift(u, 1)
    u₊₁ = circshift(u, -1)
    u₊₂ = circshift(u, -2)
    error()
    @. -(-u₊₂ + 8u₊₁ - 8u₋₁ + u₋₂) / 12Δx
end

function (e::Diffusion{6})(u, p, t)
    Δx = e.l / size(u, 1)
    u₋₃ = circshift(u, 3)
    u₋₂ = circshift(u, 2)
    u₋₁ = circshift(u, 1)
    u₊₁ = circshift(u, -1)
    u₊₂ = circshift(u, -2)
    u₊₃ = circshift(u, -3)
    error()
    @. -(-u₊₂ + 8u₊₁ - 8u₋₁ + u₋₂) / 12Δx
end

function (e::Diffusion{8})(u, p, t)
    Δx = e.l / size(u, 1)
    u₋₄ = circshift(u, 3)
    u₋₃ = circshift(u, 3)
    u₋₂ = circshift(u, 2)
    u₋₁ = circshift(u, 1)
    u₊₁ = circshift(u, -1)
    u₊₂ = circshift(u, -2)
    u₊₃ = circshift(u, -3)
    u₊₄ = circshift(u, -3)
    error()
    @. -(-u₊₂ + 8u₊₁ - 8u₋₁ + u₋₂) / 12Δx
end

eqname(::Diffusion) = "diffusion"
