module ConvolutionalClosure

using FFTW
using LinearAlgebra
using Lux
using Optimisers
using OrdinaryDiffEq
using Plots
using Random
using SparseArrays
using SpecialFunctions
using Zygote

export top_hat, gaussian, apply_matrix
export Convection,
    Diffusion, Burgers, KortewegDeVries, KuramotoSivashinsky, Schrodinger, eqname
export relerr, trajectory_loss, derivative_loss
export plotsol, plotfield, plotmat
export create_pod
export create_data
export solve_equation, solve_matrix, rk4
export glorot_uniform_Float64, relu, gelu
export FourierLayer, MatrixTransformLayer
export convolutional_closure,
    convolutional_matrix_closure, fourier_closure, matrix_transform_closure
export train
export circulant, constant_interpolator, linear_interpolator
export extend1D, extend2D
export transpose_kernel, apply_stencils, apply_stencils_transpose, apply_stencils_nonsquare

include("filters.jl")
include("equations/equations.jl")
include("solvers.jl")
include("loss.jl")
include("plots.jl")
include("pod.jl")
include("create_data.jl")
include("init.jl")
include("fourier_layer.jl")
include("closure_models.jl")
include("train.jl")
include("utils.jl")

end
