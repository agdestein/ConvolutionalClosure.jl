function create_pod(f, u, nmode = size(u, 1))
    (; U, S, V) = svd(reshape(Array(u), size(u, 1), :))
    Φ = U[:, 1:nmode]
    pod(u, p, t) = Φ' * f(Φ * u, p, t)
end
