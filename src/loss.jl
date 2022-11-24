"""
    relerr(u, v, t)

Average relative errors.
"""
function relerr(u, v, t)
    n = size(u, 1)
    u = reshape(u, n, :)
    v = reshape(v, n, :)
    k = size(u, 2)
    sum(norm(u[:, i] - v[:, i]) / norm(v[:, i]) for i ∈ 1:k) / k
end

"""
Compute trajectory-fitting loss.
Chooses a random subset of the solutions and time points at each evaluation.
Note that `u` is of size `(nx, nsample, ntime)`
"""
function create_loss_trajectory_fit(
    f,
    p,
    u,
    t;
    nsample = size(u, 2),
    ntime = length(t),
    λ = 0,
    kwargs...,
)
    iu = Zygote.@ignore sort(shuffle(1:size(u, 2))[1:nsample])
    it = Zygote.@ignore sort(shuffle(1:length(t))[1:ntime])
    u = u[:, iu, it]
    t = t[it]
    sol = solve_equation(f, u[:, :, 1], p, t; kwargs...)
    # data = sum(abs2, sol - u) / length(u)
    data = sum(abs2, sol - u) / sum(abs2, u)
    reg = sum(abs2, p) / length(p)
    data + λ * reg
end

"""
Compute derivative-fitting loss.
Chooses a random subset (`nuse`) of the data samples at each evaluation.
Note that both `u` and `dudt` are of size `(nx, nsample)`.
"""
function loss_derivative_fit(f, p, dudt, u; nuse = size(u, 2), λ = 0)
    i = Zygote.@ignore sort(shuffle(1:size(u, 2))[1:nuse])
    dudt = dudt[:, i]
    u = u[:, i]
    predict = f(u, p, zero(eltype(u)))
    # data = sum(abs2, predict - dudt) / length(u)
    data = sum(abs2, predict - dudt) / sum(abs2, dudt)
    reg = sum(abs2, p) / length(p)
    data + λ * reg
end
