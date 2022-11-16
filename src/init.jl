"""
Generate random weights as in `Lux.glorot_uniform`, but with Float64.
https://github.com/avik-pal/Lux.jl/blob/51bbf8dc489155c53f5f034b636848bdaabfc55d/src/utils.jl#L45-L48
"""
function glorot_uniform_Float64(rng::AbstractRNG, dims::Integer...; gain::Real = 1)
    scale = gain * sqrt(24 / sum(Lux._nfan(dims...)))
    return (rand(rng, dims...) .- 1 / 2) .* scale
end
