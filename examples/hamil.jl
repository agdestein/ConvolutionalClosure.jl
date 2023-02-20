using Distributions
using ForwardDiff
using GLMakie
using LinearAlgebra
using OrdinaryDiffEq

H(x) = 1 / 2 * x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2 + x[1]^2 * x[3]^2

f(u, p, t) = [u[2]; -u[1] * (1 + u[3]^2); u[4]; -u[3] * (1 + u[1]^2)]

R(x) = f(x, nothing, nothing)

function f!(du, u, p, t)
    du[1] = u[2]
    du[2] = -u[1] * (1 + u[3]^2)
    du[3] = u[4]
    du[4] = -u[3] * (1 + u[1]^2)
end

f1(u, p, t) = [u[2]; -u[1] * (1 + 1 / (1 + u[1]^2))]
ft(u, p, t) = [u[2]; -u[1] * (1 + 1 / (1 + u[1]^2)) - 2t * u[1]^2 * u[2] / (1 + u[1]^2)^2]
fr(u, p, t) = [u[2]; -u[1]]

function s(f, x, t; abstol = 1e-8, reltol = 1e-4, kwargs...)
    prob = ODEProblem(f, x, extrema(t), nothing)
    solve(prob, Tsit5(); saveat = t, abstol, reltol, kwargs...)
end

ϕ(x, t) = s(f, x, [0.0, t]; abstol = 1e-8, reltol = 1e-6)[end]

x = [1, 0, 0.5, 0.5]
t = LinRange(0, 20, 500)
sol = s(f, x, t)

lines(t, sol[1, :])
lines!(t, sol[2, :])

Σprime(x) = [1/(1+x[1]^2) 0; 0 1]
xprime(xbar) = rand(MvNormal(Σprime(xbar)))

scatter(
    [Point2f(xprime([1.0, 0.0])) for _ = 1:10000];
    axis = (;
        # aspect = DataAspect(),
        xlabel = "x₃",
        ylabel = "x₄",
    ),
)

save(loc * "xprime.pdf", current_figure())

xbar = [1.0, 0.0]
n = 10_000
umean = sum(x -> s(f!, x, t), [xbar; xprime(xbar)] for _ = 1:n) / n

fig, ax, = lines(t, umean[1, :]; label = "u₁", axis = (; xlabel = "t", ylabel = "E[u]"))
lines!(ax, t, umean[2, :], label = "u₂")
axislegend()

save(loc * "truemean.pdf", current_figure())

lines(t, umean[1, :])
lines(t, umean[2, :])

lines(umean[1, :], umean[2, :])

# Compare models for ubar
u1 = s(f1, xbar, t)
ut = s(ft, xbar, t)
ur = s(fr, xbar, t)

fig, ax, = lines(t, umean[1, :]; label = "True mean", axis = (; xlabel = "t", ylabel = "E[u₁]"))
lines!(ax, t, u1[1, :]; label = "R̄ ")
lines!(ax, t, ut[1, :]; label = "R̄ + t-model")
lines!(ax, t, ur[1, :]; label = "Galerkin")
axislegend()

save(loc * "Rbar.pdf", current_figure())
save(loc * "tmodel.pdf", current_figure())
save(loc * "galerkin.pdf", current_figure())

# Check right hand side
x = [1.0, 0.0, 0.5, 0.5]
t = 10.0
u = ϕ(x, t)

fx = R(x)
fu = R(u)

dudx = ForwardDiff.jacobian(x -> ϕ(x, t), x)

dudx * R(x)
R(u)

dudx * R(x) - R(u)


