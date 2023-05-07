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
    spectral_weights = init_weight(rng, n, n, kmax + 1, 2),
)
Lux.initialstates(::AbstractRNG, ::FourierLayer) = (;)
Lux.parameterlength((; kmax, n)::FourierLayer) = n * n + n * n * (kmax + 1) * 2
Lux.statelength(::FourierLayer) = 0

# Pass inputs through Fourier layer
function ((; n, kmax, σ)::FourierLayer)(x, params, state)
    _, nx, _ = size(x)
    @assert kmax ≤ nx "Fourier layer input must be discretized on at least `kmax` points"

    # Destructure params
    W = params.spatial_weight
    R = params.spectral_weights
    R = R[:, :, :, 1] .+ im .* R[:, :, :, 2]

    # Spatial part (applied point-wise)
    y = reshape(x, n, :)
    y = W * y
    y = reshape(y, n, nx, :)

    # Spectral part (applied mode-wise)
    # TODO: Check normalization of `fft` for different discretizations
    z = fft(x, 2)
    z = z[:, 1:kmax + 1, :]
    # z1 = z[:, 1:kmax, :]
    # z2 = z[:, end-kmax+1:end, :]
    # z = [z1 z2]
    z = reshape(z, 1, n, kmax + 1, :)
    z = sum(R .* z; dims = 2)
    z = reshape(z, n, kmax + 1, :)
    # z = [z[:, 1:kmax, :] zeros(n, nx - 2 * kmax, nsample) z[:, kmax+1:end, :]]
    z = [z[:, 1:kmax + 1, :] zeros(n, nx - kmax - 1, nsample)]
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

"""
    MatrixTransformLayer(T, n; σ = identity, init_weight = glorot_uniform_Float64)

Matrix transform layer operating on uniformly discretized functions with `n`
pointwise output values. Unlike [`FourierLayer`](@ref), this layer is bound to
a particular discretization through the precomputed transform matrix. Inputs
and outputs are both arrays of size `(n, nx, nsample)`, where `nx = size(T, 1)`,
and the `nsample` samples are treated independently.
"""
Base.@kwdef struct MatrixTransformLayer{M,A,F} <: Lux.AbstractExplicitLayer
    T::M
    n::Int
    σ::A = identity
    init_weight::F = glorot_uniform_Float64
end

Lux.initialparameters(rng::AbstractRNG, (; T, n, init_weight)::MatrixTransformLayer) = (;
    spatial_weight = init_weight(rng, n, n),
    spectral_weights = init_weight(rng, n, n, size(T, 2)),
)
Lux.initialstates(::AbstractRNG, ::MatrixTransformLayer) = (;)
Lux.parameterlength((; T, n)::MatrixTransformLayer) = n * n + n * n * size(T, 2)
Lux.statelength(::MatrixTransformLayer) = 0

# Pass inputs through matrix transform layer
function ((; T, n, σ)::MatrixTransformLayer)(x, params, state)
    nx, nmode = size(T)

    # Destructure params
    W = params.spatial_weight
    R = params.spectral_weights

    # Spatial part (applied point-wise)
    y = reshape(x, n, :)
    y = W * y
    y = reshape(y, n, nx, :)

    # Spectral part (applied mode-wise)
    z = reshape(x, n, 1, nx, :)
    z = reshape(T', 1, nmode, nx) .* z
    z = sum(z; dims = 3)
    z = reshape(z, 1, n, nmode, :)
    z = sum(R .* z; dims = 2)
    z = reshape(z, n, 1, nmode, :)
    z = reshape(T, 1, nx, nmode) .* z
    z = sum(z; dims = 3)
    z = reshape(z, n, nx, :)

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
