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
Apply filter to a collection of solutions.
"""
function apply_filter(W, u)
    ū = zeros(eltype(u), size(W, 1), size(u, 2), size(u, 3))
    for i ∈ 1:size(u, 3)
        @views mul!(ū[:, :, i], W, u[:, :, i])
    end
    ū
end
