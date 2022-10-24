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
