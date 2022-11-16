"""
    KuramotoSivashinsky(l, N = 2)

Kuramoto-Sivashinsky equation with domain length `l` and order `N`.
"""
struct KuramotoSivashinsky{N,T} <: AbstractEquation
    l::T
    KuramotoSivashinsky(l, N = 2) = new{N,typeof(l)}(l)
end

"""
Compute right hand side of Kuramoto-Sivashinsky equation.
This works for both vector and matrix `u` (one or many solutions).
"""
function (e::KuramotoSivashinsky{2})(u, p, t)
    Δx = e.l / size(u, 1)
    u₋₂ = circshift(u, 2)
    u₋₁ = circshift(u, 1)
    u₊₁ = circshift(u, -1)
    u₊₂ = circshift(u, -2)
    du = @. -(u₊₁ - 2u + u₋₁) / Δx^2 - (u₊₂ - 4u₊₁ + 6u - 4u₋₁ + u₋₂) / Δx^4 -
       (u₊₁^2 - u₋₁^2) / 4Δx
    du
end

eqname(::KuramotoSivashinsky) = "kuramoto_sivashinsky"
