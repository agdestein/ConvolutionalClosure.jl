module ConvolutionalClosure

using LinearAlgebra

export top_hat, gaussian, apply_filter
export Convection, Burgers, KortewegDeVries, KuramotoSivashinsky
export relerr, loss_embedded, loss_derivative_fit

include("filters.jl")
include("equations/equations.jl")
include("loss.jl")

end
