"""
    solve_equation(f, u₀, p, t; kwargs...)

Solve equation from `t[1]` to `t[end]`.
The ODE right hand side `f` should work on one vector with size `(N, 1)` and with multiple vectors in parallel in in a matrix of size `(N, nvector)`.
The initial conditions `u₀` are of size `N × n_IC`.
The solution is saved at `t` of size `n_t`.

Return an `ODESolution` acting as an array of size `N × n_IC × n_t`.
"""
function solve_equation(f, u₀, p, t; kwargs...)
    problem = ODEProblem(f, u₀, extrema(t), p)
    solve(problem, Tsit5(); saveat = t, kwargs...)
end

apply_mat(u, p, t) = p * u
function solve_matrix(A, u₀, t, solver = Tsit5(); kwargs...)
    problem = ODEProblem(apply_mat, u₀, extrema(t), A)
    # problem = ODEProblem(DiffEqArrayOperator(A), u₀, extrema(t))
    solve(problem, solver; saveat = t, kwargs...)
end

"""
    rk4(f, p, u₀, t, Δt)

Solve ODE ``du/dt = f(u, p, t)`` with RK4, where ``p`` are parameters.
"""
function rk4(f, u₀, p, t, Δt)
    u = u₀
    nt = length(t)
    sol = reshape(u₀, size(u₀)..., 1)
    for i = 1:nt-1
        n = round(Int, (t[i+1] - t[i]) / Δt)
        Δt = (t[i+1] - t[i]) / n
        tᵢ = t[i]
        for _ = 1:n
            k₁ = f(u, p, tᵢ)
            k₂ = f(u + Δt / 2 * k₁, p, tᵢ + Δt / 2)
            k₃ = f(u + Δt / 2 * k₂, p, tᵢ + Δt / 2)
            k₄ = f(u + Δt * k₃, p, t + Δt)
            u = @. u + Δt * (k₁ / 6 + k₂ / 3 + k₃ / 3 + k₄ / 6)
            tᵢ = tᵢ + Δt
        end
        sol = cat(sol, u; dims = 3)
    end
    sol
end
