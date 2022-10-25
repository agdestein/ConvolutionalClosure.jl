"""
    Burgers(l, N = 2)

Burgers equation with domain length `l`, viscosity `viscosity` and order `N`.
"""
struct Burgers{N,T} <: AbstractEquation
    l::T
    viscosity::T
    Burgers(l, viscosity, N = 2) = new{N,typeof(l)}(l, viscosity)
end

"""
    (::Burgers)(u, p, t)

Compute right hand side of Burgers equation.
This works for both vector and matrix `u` (one or many solutions).
"""
function (::Burgers) end

function (e::Burgers{2})(u, p, t)
    Δx = e.l / size(u, 1)
    μ = e.viscosity
    u₋ = circshift(u, 1)
    u₊ = circshift(u, -1)
    du = @. -(u₊^2 - u₋^2) / 4Δx + μ * (u₊ - 2u + u₋) / Δx^2
    du
end

function (e::Burgers{4})(u, p, t)
    Δx = e.l / size(u, 1)
    μ = e.viscosity
    u₋₂ = circshift(u, 2)
    u₋₁ = circshift(u, 1)
    u₊₁ = circshift(u, -1)
    u₊₂ = circshift(u, -2)
    du = @. (u₊₂ / 12 - 8u₊₁ + 8u₋₁ - u₋₂ / 12) / 12Δx +
       μ * (-u₊₂^2 / 12 + 16u₊₁^2 - 30u^2 + 16u₋₁^2 - u₋₂^2 / 12) / 12Δx^2
    du
end

# function (e::Burgers{4})(u, p, t)
#     Δx = e.l / size(u, 1)
#     μ = e.viscosity
#     u₋ = circshift(u, 1)
#     u₊ = circshift(u, -1)
#     # du = @. -(u₊^2 - u₋^2) / 4Δx + μ * (u₋ - 2u + u₊) / Δx^2
#     # du = @. -(u₊^2 + u₊ * u - u * u₋ - u₋^2) / 6Δx + μ * (u₋ - 2u + u₊) / Δx^2
#     a₊ = u₊ + u
#     a₋ = u + u₋
#     du = @. -((a₊ < 0)u₊^2 + (a₊ > 0)u^2 - (a₋ < 0)u^2 - (a₋ > 0)u₋^2) / Δx + μ * (u₋ - 2u + u₊) / Δx^2
#     du
# end
