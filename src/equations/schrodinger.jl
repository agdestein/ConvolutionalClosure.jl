"""
    Schrodinger(l, N = 2)

Schrodinger equation with domain length `l`, viscosity `viscosity` and order `N`.
"""
struct Schrodinger{N} <: AbstractEquation
    Schrodinger(N = 2) = new{N}()
end

"""
    (::Schrodinger)(u, p, t)

Compute right hand side of Schrodinger equation.
This works for both vector and matrix `u` (one or many solutions).
"""
function (::Schrodinger) end

function (e::Schrodinger{2})(u, p, t)
    Δx = 2π / size(u, 1)
    u₋ = circshift(u, 1)
    u₊ = circshift(u, -1)
    du = @. im * (u₊ - 2u + u₋) / Δx^2 - im / 4 * (3 * abs(u)^2 * u + conj(u)^3)
    du
end
