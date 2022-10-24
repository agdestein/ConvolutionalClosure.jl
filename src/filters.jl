top_hat(Δ, x) = (abs(x) ≤ Δ / 2) / Δ
gaussian(Δ, x) = √(6 / π) / Δ * exp(-6x^2 / Δ^2)

"""
Apply filter to a collection of solutions.
"""
function apply_filter(W, u)
    ū = similar(u)
    for i ∈ 1:size(u, 3)
        @views mul!(ū[:, :, i], W, u[:, :, i])
    end
    ū
end
