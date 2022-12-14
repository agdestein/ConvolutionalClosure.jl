module ConvolutionalClosure

using LinearAlgebra
using Lux
using Optimisers
using OrdinaryDiffEq
using Plots
using Random
using SparseArrays
using Zygote

export top_hat, gaussian, apply_filter
export Convection, Diffusion, Burgers, KortewegDeVries, KuramotoSivashinsky, eqname
export rk4
export relerr, trajectory_loss, derivative_loss
export plotsol, plotmat
export create_pod
export create_data
export solve_equation
export glorot_uniform_Float64
export convolutional_closure, convolutional_matrix_closure
export train
export circulant, interpolation_matrix

include("filters.jl")
include("equations/equations.jl")
include("solvers.jl")
include("loss.jl")
include("plots.jl")
include("pod.jl")
include("create_data.jl")
include("solve_equation.jl")
include("init.jl")
include("closure_models.jl")
include("train.jl")
include("utils.jl")

end
