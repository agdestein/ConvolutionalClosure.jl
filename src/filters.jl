"""
    top_hat(Δ, x)

Top-hat filter weight at point `x` for filter width `Δ`.
"""
top_hat(Δ, x) = (abs(x) ≤ Δ / 2) / Δ

"""
    gaussian(Δ, x)

Gaussian filter weight at point `x` for filter width `Δ`.
"""
gaussian(Δ, x) = √(6 / π) / Δ * exp(-6x^2 / Δ^2)

"""
    gaussian(Δx, Δy, x, y)

Gaussian filter weight at point `(x, y)` for filter width `(Δx, Δy)`.
"""
gaussian(Δx, Δy, x, y) = 6 / (π * Δx * Δy) * exp(-6x^2 / Δx^2 - 6y^2 / Δy^2)


"""
    apply_filter(W, u)

Apply filter to a collection of solutions.
"""
function apply_filter(W, u)
    u = Array(u)
    s = size(u)
    u = reshape(u, s[1], :)
    y = W * u
    reshape(y, size(y, 1), s[2:end]...)
end
