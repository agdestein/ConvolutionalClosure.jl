"""
    FourierLayer(n, kmax; σ = identity, init_weight = glorot_uniform_Float64)

Fourier layer operating on uniformly discretized functions with `n` pointwise
output values. Inputs and outputs are both arrays of size `(n, nx, nsample)`, where `nx` is
any number of discrete uniform spatial points that is higher than the cut-off
frequency `kmax`, and the `nsample` samples are treated independently.
"""
Base.@kwdef struct FourierLayer{A,F} <: Lux.AbstractExplicitLayer
    n::Int
    kmax::Int
    σ::A = identity
    init_weight::F = glorot_uniform_Float64
end

Lux.initialparameters(rng::AbstractRNG, (; kmax, n, init_weight)::FourierLayer) = (;
    spatial_weight = init_weight(rng, n, n),
    spectral_weights_real = init_weight(rng, n, n, kmax),
    spectral_weights_imag = init_weight(rng, n, n, kmax),
)
Lux.initialstates(::AbstractRNG, ::FourierLayer) = (;)
Lux.parameterlength((; kmax, n)::FourierLayer) = n * n + 2 * n * n * kmax
Lux.statelength(::FourierLayer) = 0

# Pass inputs through Fourier layer
function ((; n, kmax, σ)::FourierLayer)(x, params, state)
    _, nx, nsample = size(x)
    @assert kmax ≤ nx "Fourier layer input must be discretized on at least `kmax` points"

    # Destructure params
    W = params.spatial_weight
    R = params.spectral_weights_real .+ im .* params.spectral_weights_imag

    # Spatial part (applied point-wise)
    y = reshape(x, n, :)
    y = W * y
    y = reshape(y, n, nx, nsample)

    # Spectral part (applied mode-wise)
    # Todo: Check normalization of `fft` for different discretizations
    z = fft(x, 2)
    z = z[:, 1:kmax, :]
    z = reshape(z, 1, n, kmax, nsample)
    z = sum(R .* z; dims = 2)
    z = reshape(z, n, kmax, nsample)
    z = hcat(z, zeros(n, nx - kmax, nsample))
    z = real.(ifft(z, 2))

    # Outer layer: Activation over combined spatial and spectral parts
    # Note: Even though high frequencies are chopped of in `z` and may
    # possibly not be present in the input at all, `σ` creates new high frequencies.
    # High frequency functions may thus be represented using a sequence of
    # Fourier layers. In this case, the `y`s are the only place where
    # information contained in high
    # input frequencies survive in a Fourier layer.
    v = σ.(y .+ z)

    # Fourier layer does not change state
    v, state
end
