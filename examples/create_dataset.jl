using OrdinaryDiffEq

"""
    create_data(f, K, x, nsolution, t; decay = k -> 1 / (1 + abs(k)), kwargs...)

Args:

  - Right hand side `f(u, p, t)`
  - Maximum frequency in initial conditions `K`
  - Spatial points `x = LinRange(0, L, N + 1)[2:end]`
  - Number of different initial conditions `nsolution`
  - Time points to save `t = LinRange(0, T, ntime)`

Kwargs:

  - Frequency decay function (for initial conditions) `decay(k)`
  - Other kwargs: Pass to ODE solver (e.g. `reltol = 1e-4`, `abstol = 1e-6`)

Returns:

  - Solution `u` of size `(nx, nsolution, ntime)`

To get initial conditions, do `u₀ = u[:, :, 1]`.
"""
function create_data(f, K, x, nsolution, t; decay = k -> 1 / (1 + abs(k)), kwargs...)
    # Domain length (assume x = LinRange(0, L, N + 1)[2:end]) for some `L` and `N`)
    L = x[end]

    # Fourier basis
    basis = [exp(2π * im * k * x / L) for x ∈ x, k ∈ -K:K]

    # Fourier coefficients with random phase and amplitude
    c = [randn() * decay(k) * exp(-2π * im * rand()) for k ∈ -K:K, _ ∈ 1:nsolution]

    # Random initial conditions (real-valued)
    u₀ = real.(basis * c)

    # Solve ODE
    problem = ODEProblem(f, u₀, extrema(t), nothing)
    sol = solve(problem, Tsit5(); saveat = t, kwargs...)

    Array(sol)
end

L() = 1.0
μ() = 0.001

function burgers(u, p, t)
    Δx = L() / size(u, 1)
    u₋ = circshift(u, 1)
    u₊ = circshift(u, -1)
    @. -(u₊^2 - u₋^2) / 4Δx + μ() * (u₊ - 2u + u₋) / Δx^2
end

N = 200
x = LinRange(0.0, L(), N + 1)[2:end]
t_train = LinRange(0.0, 0.1, 60)
t_test = LinRange(0.0, 1.0, 30)

K = 100

u_train = create_data(burgers, K, x, 500, t_train; reltol = 1e-4, abstol = 1e-6)
u_test = create_data(burgers, K, x, 50, t_test; reltol = 1e-4, abstol = 1e-6)
