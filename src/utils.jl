"""
    circulant(n, inds, stencil)

Create circulant `SparseMatrixCSC`.
"""
circulant(n, inds, stencil) = spdiagm(
    (i => fill(s, n - abs(i)) for (i, s) ∈ zip(inds, stencil))...,
    (i - sign(i) * n => fill(s, abs(i)) for (i, s) ∈ zip(inds, stencil))...,
)

"""
    interpolation_matrix(x, y)

Create matrix for interpolating from grid ``x \\in \\mathbb{R}^N`` to ``y \\in \\mathbb{R}^M``.
"""
function interpolation_matrix(L, x, y)
    @assert issorted(x)
    @assert issorted(y)
    @assert x[end] - x[1] ≤ L
    @assert y[end] - y[1] ≤ L
    N = length(x)
    M = length(y)
    a = [x[end] - L; x[1:end]]'
    b = [x[1:end]; x[1] + L]'
    Ia = a .< y .≤ b
    Ib = circshift(Ia, (0, 1))
    A = @. (b - y) / (b - a)
    B = circshift(@.((y - a) / (b - a)), (0, 1))
    P = @. A * Ia + B * Ib
    IP = P[:, 2:end]
    IP[:, end] += P[:, 1]
    sparse(IP)
end
