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
function (kdv::KuramotoSivashinsky{2})(u, p, t)
    Δx = kdv.l / size(u, 1)
    u₋₂ = circshift(u, 2)
    u₋₁ = circshift(u, 1)
    u₊₁ = circshift(u, -1)
    u₊₂ = circshift(u, -2)
    error("Not implemented")
    du = @. (u₊₂ - 2u₊₁ + 2u₋₁ - u₋₂) / 2Δx^3 - 6 * u * (u₊₁ - u₋₁) / 2Δx
    du
end
