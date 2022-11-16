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
