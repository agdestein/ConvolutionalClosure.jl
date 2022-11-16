"""
    create_data(x, K, nsample; decay = Returns(1))

Create `nsample` random signals on `x` with maximum frequency `K` and frequency decay
`decay(k)`.
"""
function create_data(x, K, nsample; decay = Returns(1))
    # Domain length
    l = x[end]

    # Fourier basis
    basis = [exp(2π * im * k * x / l) for x ∈ x, k ∈ -K:K]

    # Fourier coefficients with random phase and amplitude
    c = [randn() * exp(-2π * im * rand()) * decay(k) for k ∈ -K:K, _ ∈ 1:nsample]

    # Random data samples (real-valued)
    real.(basis * c)
end
