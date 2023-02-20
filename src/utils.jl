"""
    circulant(n, inds, stencil)

Create circulant `SparseMatrixCSC`.
"""
circulant(n, inds, stencil) = spdiagm(
    (i => fill(s, n - abs(i)) for (i, s) ∈ zip(inds, stencil))...,
    (i - sign(i) * n => fill(s, abs(i)) for (i, s) ∈ zip(inds, stencil))...,
)

"""
    extend1D(u, r)

Extend the vector `u` periodically with an overlap of `r`.
"""
function extend1D(u, r)
    s, ssupp... = size(u)
    u = reshape(u, s, :)
    u = [
        u[end-r+1:end, :]
        u
        u[1:r, :]
    ]
    reshape(u, s + 2r, ssupp...)
end

"""
    extend2D(u, rx, ry = rx)

Extend the matrix `u` periodically with an overlap of `(rx, ry)`.
"""
function extend2D(u, rx, ry = rx)
    sx, sy, ssupp... = size(u)
    u = reshape(u, sx, sy, :)
    u = [
        u[end-rx+1:end, end-ry+1:end, :] u[:, end-ry+1:end, :] u[1:rx, end-ry+1:end, :]
        u[end-rx+1:end, :, :] u u[1:rx, :, :]
        u[end-rx+1:end, 1:ry, :] u[:, 1:ry, :] u[1:rx, 1:ry, :]
    ]
    reshape(u, sx + 2rx, sy + 2ry, ssupp...)
end

"""
    constant_interpolator(x, y)

Create matrix for interpolating from grid ``x \\in \\mathbb{R}^N`` to ``y \\in \\mathbb{R}^M``.
This function uses a zeroth order interpolation, i.e. the function is piece-wise constant.
"""
function constant_interpolator(L, x, y)
    @assert issorted(x)
    @assert issorted(y)
    @assert x[end] - x[1] ≤ L
    @assert y[end] - y[1] ≤ L
    a = [x[end] - L; x]
    b = [x; x[1] + L]
    m = (a + b) / 2
    m = [m[end-1] - L; m]
    A = m[1:end-1]' .< y .≤ m[2:end]'
    P = A[:, 2:end]
    P[:, end] += A[:, 1]
    sparse(P)
end

"""
    linear_interpolator(x, y)

Create matrix for interpolating from grid ``x \\in \\mathbb{R}^N`` to ``y \\in \\mathbb{R}^M``.
"""
function linear_interpolator(L, x, y)
    @assert issorted(x)
    @assert issorted(y)
    @assert x[end] - x[1] ≤ L
    @assert y[end] - y[1] ≤ L
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
