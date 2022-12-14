"""
    relerr(u, v)

Average relative errors along first dimension.
"""
function relerr(u, v)
    n = size(u, 1)
    u = reshape(u, n, :)
    v = reshape(v, n, :)
    k = size(u, 2)
    sum(norm(u[:, i] - v[:, i]) / norm(v[:, i]) for i ∈ 1:k) / k
end

"""
Compute trajectory-fitting loss.
Chooses a random subset of the solutions and time points at each evaluation.
Note that `u` is of size `(nx, nsolution, ntime)`
"""
function trajectory_loss(
    f,
    p,
    u,
    t;
    nsolution = size(u, 2),
    ntime = length(t),
    λ = 0,
    kwargs...,
)
    iu = Zygote.@ignore sort(shuffle(1:size(u, 2))[1:nsolution])
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
function derivative_loss(f, p, dudt, u; nuse = size(u, 2), λ = 0)
    nsample = size(u)[end]
    d = ndims(u)
    i = Zygote.@ignore sort(shuffle(1:nsample)[1:nuse])
    dudt = selectdim(dudt, d, i)
    u = selectdim(u, d, i)
    predict = f(u, p, zero(eltype(u)))
    # data = sum(abs2, predict - dudt) / length(u)
    data = sum(abs2, predict - dudt) / sum(abs2, dudt)
    reg = sum(abs2, p) / length(p)
    data + λ * reg
end
