"""
    KortewegDeVries(l, N = 2)

Korteweg-De Vries equation with domain length `l` and order `N`.
"""
struct KortewegDeVries{N,T} <: AbstractEquation
    l::T
    KortewegDeVries(l, N = 2) = new{N,typeof(l)}(l)
end

"""
Compute right hand side of Korteweg-De Vries equation.
This works for both vector and matrix `u` (one or many solutions).
"""
function (e::KortewegDeVries{2})(u, p, t)
    Δx = e.l / size(u, 1)
    u₋₂ = circshift(u, 2)
    u₋₁ = circshift(u, 1)
    u₊₁ = circshift(u, -1)
    u₊₂ = circshift(u, -2)
    @. (u₊₂ - 2u₊₁ + 2u₋₁ - u₋₂) / 2Δx^3 - 3 * (u₊₁^2 - u₋₁^2) / 2Δx
    # @. (u₊₂ - 2u₊₁ + 2u₋₁ - u₋₂) / 2Δx^3 - 6 * u * (u₊₁ - u₋₁) / 2Δx
end

eqname(::KortewegDeVries) = "korteweg_de_vries"
