"""
    relerr(u, v, t)

Average relative errors.
"""
function relerr(u, v, t)
    sum(
        norm(u[:, j, i] - v[:, j, i]) / norm(v[:, j, i]) for i ∈ 1:size(u, 3),
        j = 1:size(u, 2)
    ) / prod(size(u)[2:3])
end

"""
    loss_embedded(f, p, u, t, λ; kwargs...)

Compute trajectory-fitting loss.
"""
function loss_embedded(f, p, u, t, λ; kwargs...)
    sol = solve_equation(f, u[:, :, 1], p, t; kwargs...)
    # data = sum(abs2, sol - u) / length(u)
    data = sum(abs2, sol - u) / sum(abs2, u)
    reg = sum(abs2, p) / length(p)
    data + λ * reg
end

"""
    loss_derivative_fit(f, p, dudt, u, λ)

Compute derivative-fitting loss.
"""
function loss_derivative_fit(f, p, dudt, u, λ)
    predict = f(u, p, zero(eltype(u)))
    # data = sum(abs2, predict - dudt) / length(u)
    data = sum(abs2, predict - dudt) / sum(abs2, dudt)
    reg = sum(abs2, p) / length(p)
    data + λ * reg
end