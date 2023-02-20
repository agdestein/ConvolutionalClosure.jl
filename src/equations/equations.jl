abstract type AbstractEquation end

"""
    eqname(equation)

Get name of equation.
"""
function eqname end

include("convection.jl")
include("diffusion.jl")
include("burgers.jl")
include("korteweg_de_vries.jl")
include("kuramoto_sivashinsky.jl")
include("schrodinger.jl")
