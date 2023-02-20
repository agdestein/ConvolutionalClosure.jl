"""
    Convection(l, N = 2)

Convection equation with domain length `l` and order `N`.
"""
struct Convection{N,T} <: AbstractEquation
    l::T
    Convection(l, N = 2) = new{N,typeof(l)}(l)
end

"""
    (::Convection)(u, p, t)

Compute right hand side of convection equation.
This works for both vector and matrix `u` (one or many solutions).
"""
function (::Convection) end

function (e::Convection{2})(u, p, t)
    Δx = e.l / size(u, 1)
    u₋₁ = circshift(u, 1)
    u₊₁ = circshift(u, -1)
    du = @. -(u₊₁ - u₋₁) / 2Δx
    du
end

function (e::Convection{4})(u, p, t)
    Δx = e.l / size(u, 1)
    u₋₂ = circshift(u, 2)
    u₋₁ = circshift(u, 1)
    u₊₁ = circshift(u, -1)
    u₊₂ = circshift(u, -2)
    du = @. -(-u₊₂ + 8u₊₁ - 8u₋₁ + u₋₂) / 12Δx
    du
end

function (e::Convection{6})(u, p, t)
    Δx = e.l / size(u, 1)
    u₋₃ = circshift(u, 3)
    u₋₂ = circshift(u, 2)
    u₋₁ = circshift(u, 1)
    u₊₁ = circshift(u, -1)
    u₊₂ = circshift(u, -2)
    u₊₃ = circshift(u, -3)
    du = @. -(u₊₃ - 9u₊₂ + 45u₊₁ - 45u₋₁ + 9u₋₂ - u₋₃) / 60Δx
    du
end

function (e::Convection{8})(u, p, t)
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
    du = @. -(-u₊₂ + 8u₊₁ - 8u₋₁ + u₋₂) / 12Δx
    du
end

eqname(::Convection) = "convection"
