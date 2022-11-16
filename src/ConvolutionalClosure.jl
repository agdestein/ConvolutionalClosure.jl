module ConvolutionalClosure

using LinearAlgebra
using Lux
using OrdinaryDiffEq
using Plots
using Random
using SparseArrays

export top_hat, gaussian, apply_filter
export Convection, Diffusion, Burgers, KortewegDeVries, KuramotoSivashinsky, eqname
export rk4
export relerr, loss_embedded, loss_derivative_fit
export plotsol, plotmat
export create_pod
export create_data
export solve_equation
export glorot_uniform_Float64

include("filters.jl")
include("equations/equations.jl")
include("solvers.jl")
include("loss.jl")
include("plots.jl")
include("pod.jl")
include("create_data.jl")
include("solve_equation.jl")
include("init.jl")

end
