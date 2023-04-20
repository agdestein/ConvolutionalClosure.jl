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
    apply_matrix(A, x)

Apply matrix `A` to a collection of vectors `x` with any additional tensor dimensions.
"""
function apply_matrix(A, x)
    x = Array(x)
    n, s... = size(x)
    x = reshape(x, n, :)
    y = A * x
    reshape(y, size(y, 1), s...)
end
