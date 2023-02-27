"""
    convolutional_closure(r, c, a; rng = Random.default_rng, channel_augmenter = identity)

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
function convolutional_closure(
    r,
    c,
    a,
    b;
    rng = Random.default_rng(),
    channel_augmenter = identity,
)
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
    p, re = Lux.destructure(params)

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
    convolutional_matrix_closure(r, c, a; rng = Random.default_rng, channel_augmenter = identity)

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
    p, re = Lux.destructure(params)

    """
    Compute closure term for given parameters `p`.
    """
    function closure end
    closure(u, p, t) = first(NN(u, re(p), state))
    closure(u::AbstractMatrix, p, t) = reshape(closure(reshape(u, size(u)..., 1), p, t), :)

    p, closure
end
