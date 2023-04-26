"""
    convolutional_closure(r, c, a, b; rng = Random.default_rng, channel_augmenter = identity)

Create closure function with

Args:

  - `r`: Kernel radii (`nlayer`)
  - `c`: Number of channels (`nlayer + 1`) (first should be equal to number of output channels in `channel_augmenter`)
  - `a`: Activation functions (`nlayer`)
  - `b`: Bias indicators (`nlayer`)

Kwargs:

  - rng: Random number generator
  - channel_augmenter: A function `u -> hcat(channel_1(u), ..., channel_n(u))`

Return `(p, closure)`, where `p` are the initial parameters and `closure(u, p, t)` computes the closure term.
"""
function convolutional_closure(
    r,
    c,
    a,
    b;
    rng = Random.default_rng(),
    channel_augmenter = identity,
)
    # Number of input channels
    nchannel = length(channel_augmenter(1))

    c = [nchannel; c]

    # Discrete closure term for filtered equation
    NN = Chain(
        # From (nx, nsample) to (nx, nchannel, nsample)
        u -> reshape(u, size(u, 1), 1, size(u, 2)),

        # Create input channels
        channel_augmenter,

        # Manual padding along spatial dimension to account for periodicity
        u -> [u[end-sum(r)+1:end, :, :]; u; u[1:sum(r), :, :]],

        # Some convolutional layers to mimic local differential operators
        (
            Conv(
                (2r[i] + 1,),
                c[i] => c[i+1],
                a[i];
                init_weight = glorot_uniform_Float64,
                bias = b[i],
            ) for i ∈ eachindex(r)
        )...,

        # From (nx, nchannel = 1, nsample) to (nx, nsample)
        u -> reshape(u, size(u, 1), size(u, 3)),

        # # Difference for momentum conservation
        # u -> circshift(u, -1) - circshift(u, 1),
    )

    # Lux.setup(rng, NN)
    params, state = Lux.setup(rng, NN)
    p, re = destructure(params)

    """
        closure(u, p, t) 

    Compute closure term for given parameters `p`.
    """
    function closure end
    closure(u, p, t) = first(NN(u, re(p), state))
    closure(u::AbstractVector, p, t) = reshape(closure(reshape(u, :, 1), p, t), :)

    p, closure
end

"""
    convolutional_matrix_closure(r, c, a, b; rng = Random.default_rng, channel_augmenter = identity)

Create closure function with

Args:

  - `r`: Kernel radii (`nlayer`)
  - `c`: Number of channels (`nlayer + 1`) (first should be equal to number of output channels in `channel_augmenter`)
  - `a`: Activation functions (`nlayer`)
  - `b`: Bias indicators (`nlayer`)

Kwargs:

  - rng: Random number generator
  - channel_augmenter: A function `u -> hcat(channel_1(u), ..., channel_n(u)`

Return `(p, closure)`, where `p` are the initial parameters and `closure(u, p, t)` computes the closure term.
"""
function convolutional_matrix_closure(
    r,
    c,
    a,
    b;
    rng = Random.default_rng(),
    channel_augmenter = identity,
)
    # Discrete closure term for filtered equation
    NN = Chain(
        # Create input channels
        channel_augmenter,

        # Manual padding along spatial dimension to account for periodicity
        u -> [u[end-sum(r)+1:end, :, :]; u; u[1:sum(r), :, :]],

        # Some convolutional layers to mimic local differential operators
        (
            Conv(
                (2r[i] + 1,),
                c[i] => c[i+1],
                a[i];
                init_weight = glorot_uniform_Float64,
                bias = b[i],
            ) for i ∈ eachindex(r)
        )...,

        # From (nx, nchannel = 1, nsample) to (nx, nsample)
        u -> reshape(u, size(u, 1), size(u, 3)),

        # # Difference for momentum conservation
        # u -> circshift(u, -1) - circshift(u, 1),
    )

    Lux.setup(rng, NN)
    params, state = Lux.setup(rng, NN)
    p, re = destructure(params)

    """
    Compute closure term for given parameters `p`.
    """
    function closure end
    closure(u, p, t) = first(NN(u, re(p), state))
    closure(u::AbstractMatrix, p, t) = reshape(closure(reshape(u, size(u)..., 1), p, t), :)

    p, closure
end

"""
    fourier_closure(n, kmax; channel_augmenter = identity, σ = gelu, [rng])

Fourier neural operator closure with `n` latent channels, wavenumber truncation at `kmax`, and actiavation function `σ`.

In addition to an intial and final point-wise lift and compression,
this model is comprised of four inner Fourier layers.

For each Fourier layer, the output is `σ(y + z)`, where

  - `y` is a spatial point-wise linear transform of the input
  - `z` is a spectral mode-wise linear transform of the truncated spectrum of
    the input

Note: Spatial inputs can have any uniform discretization, but the number of discretization points must be at least `kmax`.
"""
function fourier_closure(
    n,
    kmax;
    channel_augmenter = identity,
    σ = gelu,
    rng = Random.default_rng(),
)
    # Number of input channels
    nchannel = length(channel_augmenter(1))

    # Discrete closure term for filtered equation
    NN = Chain(
        # From (nx, nsample) to (nchannel, nx, nsample)
        u -> reshape(u, 1, size(u, 1), size(u, 2)),

        # Augment channels
        channel_augmenter,

        # Lift input channels to latent space
        Dense(nchannel => n; bias = false),

        # Fourier layers
        FourierLayer(; n, kmax, σ),
        FourierLayer(; n, kmax, σ),
        FourierLayer(; n, kmax, σ),
        FourierLayer(; n, kmax),

        # Compress to single output channel
        Dense(n => 2n, σ),
        Dense(2n => 1; bias = false),

        # From (nchannel = 1, nx, nsample) to (nx, nsample)
        u -> reshape(u, size(u, 2), size(u, 3)),

        # # Difference for momentum conservation
        # u -> circshift(u, -1) - circshift(u, 1),
    )

    # Create initial parameters and (empty) state
    params, state = Lux.setup(rng, NN)
    p, re = destructure(params)

    """
        closure(u, p, t) 

    Compute closure term at state `u` and time `t` for given parameters `p`.
    """
    function closure end

    # Note: This method stores `NN`, `re` and `state`
    closure(u, p, t) = first(NN(u, re(p), state))

    # Make sure that single-sample vectors preserve shape
    closure(u::AbstractVector, p, t) = reshape(closure(reshape(u, :, 1), p, t), :)

    p, closure
end

"""
    matrix_transform_closure(T, n; channel_augmenter = identity, σ = gelu, [rng])

Matrix transform operator closure with transform `T`, `n` latent channels, and actiavation function `σ`.

In addition to an intial and final point-wise lift and compression,
this model is comprised of four inner matrix transform layers.

For each matrix transform layer, the output is `σ(y + z)`, where

  - `y` is a spatial point-wise linear transform of the input
  - `z` is a spectral mode-wise linear transform of the truncated spectrum of
    the input

Note: Spatial inputs must be uniformly discretized on `size(T, 1)` points.
"""
function matrix_transform_closure(
    T,
    n;
    channel_augmenter = identity,
    σ = gelu,
    rng = Random.default_rng(),
)
    # Number of input channels
    nchannel = length(channel_augmenter(1))

    # Discrete closure term for filtered equation
    NN = Chain(
        # From (nx, nsample) to (nchannel, nx, nsample)
        u -> reshape(u, 1, size(u, 1), size(u, 2)),

        # Augment channels
        channel_augmenter,

        # Lift input channels to latent space
        Dense(nchannel => n; bias = false),

        # Fourier layers
        MatrixTransformLayer(; T, n, σ),
        MatrixTransformLayer(; T, n, σ),
        MatrixTransformLayer(; T, n, σ),
        MatrixTransformLayer(; T, n),

        # Compress to single output channel
        Dense(n => 2n, σ),
        Dense(2n => 1; bias = false),

        # From (nchannel = 1, nx, nsample) to (nx, nsample)
        u -> reshape(u, size(u, 2), size(u, 3)),
    )

    # Create initial parameters and (empty) state
    params, state = Lux.setup(rng, NN)
    p, re = destructure(params)

    """
        closure(u, p, t) 

    Compute closure term at state `u` and time `t` for given parameters `p`.
    """
    function closure end

    # Note: This method stores `NN`, `re` and `state`
    closure(u, p, t) = first(NN(u, re(p), state))

    # Make sure that single-sample vectors preserve shape
    closure(u::AbstractVector, p, t) = reshape(closure(reshape(u, :, 1), p, t), :)

    p, closure
end
