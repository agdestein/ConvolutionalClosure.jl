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
    trajectory_loss(
        f,
        p,
        u,
        t;
        nsolution = size(u, ndims(u) - 1),
        ntime = length(t),
        λ = 0,
        kwargs...,
    )

Compute trajectory-fitting loss.
Chooses a random subset of the solutions and time points at each evaluation.
Note that `u` is of size `(nx, nsolution, ntime)`
"""
function trajectory_loss(
    f,
    p,
    u,
    t;
    nsolution = size(u, ndims(u) - 1),
    ntime = length(t),
    λ = 0,
    kwargs...,
)
    d = ndims(u)
    iu = Zygote.@ignore sort(shuffle(1:size(u, d - 1))[1:nsolution])
    it = Zygote.@ignore sort(shuffle(1:length(t))[1:ntime])
    # u = u[:, iu, it]
    u = selectdim(selectdim(u, d, it), d - 1, iu)
    t = t[it]
    u₀ = Array(selectdim(u, d, 1))
    sol = solve_equation(f, u₀, p, t; kwargs...)
    # data = sum(abs2, sol - u) / length(u)
    data = sum(abs2, sol - u) / sum(abs2, u)
    reg = sum(abs2, p) / length(p)
    data + λ * reg
end

"""
    derivative_loss(f, p, dudt, u; nuse = size(u, 2), λ = 0)

Compute derivative-fitting loss.
Chooses a random subset (`nuse`) of the data samples at each evaluation.
Note that both `u` and `dudt` are of size `(sample_size..., nsample)`.
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
