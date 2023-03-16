if isdefined(@__MODULE__, :LanguageServer)
    include("svd.jl")
end

plotmat(W; title = "W")

plotmat(W .!= 0; title = "Discrete filter (sparsity)")

plot(W[10, :]; yscale = :log10, ylims = (1e-8, 1))
plot(W[10, :])

plotmat(R)

plotmat(W * W')
plotmat(W' * W)
plotmat(R * W)
plotmat(W * R)

k = 9
# for k = M:-1:1
for k = 1:M
    # ind = k:M
    ind = 1:k
    # ind = k:k
    Wk = Φ[:, ind] * Diagonal(σ[ind]) * Ψ[:, ind]'
    display(plotmat(
        Wk;
        title = "k = $k",
        # clims = extrema(W),
    ))
    sleep(0.1)
end

i = [1:7; 10; 20; 30; 40; 50]
indind = k -> 1:k
# indind = k -> k:k
plot(
    (
        plotmat(
            Φ[:, indind(k)] * Diagonal(σ[indind(k)]) * Ψ[:, indind(k)]';
            xticks = iplot ≤ 4,
            yticks = iplot ∈ [1, 5, 9],
            # xlabel = "x",
            colorbar = false,
            title = k,
        ) for (iplot, k) in enumerate(i)
    )...;
    # layout = (2, 1),
    # title = "Eigenvectors of I/N - W'W/M",
    # size = (1200, 800),
    size = (800, 500),
    # plot_title = "Truncated filter matrix",
)

# savefig(loc * "nonuniform/Wk.pdf")
# savefig(loc * "nonuniform/Wkall.pdf")

plotmat(V)
plotmat(P)
plotmat(P * V)
plotmat(Ψ)

i = 20;
plot(y, Ψ[:, i]; title = "Right singular vectors $i", xlabel = "x");
i = 316;
plot(y, P[:, i]; title = "Right singular vectors $i", xlabel = "x");

for i = 1:M
    display(plot(y, Ψ[:, i]; title = "Right singular vectors $i", xlabel = "x"))
    sleep(0.1)
end

for k = 300:400
    pl = plot(y, P[:, k]; legend = false, title = "ξₖ, k = $k", xlabel = "x")
    vspan!(pl, [0.0, M / N]; fillalpha = 0.1, color = 1)
    display(pl)
    sleep(0.05)
end

i = [1:7; 10; 20; 30; 40; 50]
plot(
    (
        plot(
            y,
            Ψ[:, i];
            # label = i',
            label = false,
            xticks = iplot ≥ 9,
            xlabel = iplot ≥ 9 ? "x" : "",
            yticks = false,
            # xlabel = "x",
            title = i,
        ) for (iplot, i) in enumerate(i)
    )...;
    # layout = (2, 1),
    # title = "Eigenvectors of I/N - W'W/M",
    # size = (1200, 800),
    size = (800, 500),
)

# savefig(loc * "nonuniform/singular_vectors.pdf")

i = [1, 50, 100, 150, 200, 250]
plot(
    (
        plot(
            y,
            Ξ[:, i];
            label = false,
            xticks = iplot ≥ 4,
            xlabel = iplot ≥ 4 ? "x" : "",
            yticks = false,
            # xlabel = "x",
            title = "k = $i",
        ) for (iplot, i) in enumerate(i)
    )...;
    # layout = (2, 1),
    # title = "Eigenvectors of I/N - W'W/M",
    # size = (1200, 800),
    plot_title = "Right singular vectors ξₖ",
    size = (800, 500),
)

# savefig(loc * "nonuniform/singular_vectors_zero.pdf")

scatter(σ; title = "Singular values", legend = false)

# savefig(loc * "nonuniform/singular_values.pdf")

k = 3
for k = 1:M
    i = 1:k
    pl = plot(; xlabel = "x", title = "Signal")
    plot!(y, u; label = "u")
    # plot!(y, R * W * u;  label = "RWu")
    plot!(y, Ψ[:, i] * Ψ[:, i]' * u; label = "ΨiΨi'u")
    plot!(y, P * V * u; label = "PVu")
    display(pl)
    sleep(0.1)
end

plot(; xlabel = "x", title = "Kinetic energy")
plot!(y, ke.(u); label = "u")
plot!(y, ke.(R * W * u); label = "RWu")

plA = plotmat(A; title = "A")
plB = plotmat(B; title = "B")
plC = plotmat(C; title = "C")
plD = plotmat(D; title = "D")

plot(plA, plB, plC, plD; size = (800, 500))

# savefig(loc * "nonuniform/mzmat.pdf")

sticks(x, A[10, :]; xlims = (0, 1))

sticks(y, W[10, :]; xlims = (0, 1))
sticks(x, W[:, 100]; xlims = (0, 1))

m = 40
plot(y, R[:, m]; xlims = (0, 1))
sticks!(x, R[s*m, :]; xlims = (0, 1))
sticks(x, R[s*m, :]; xlims = (0, 1))

for m in [1:M; 1:M]
    display(plot(y, R[:, m]; xlims = (0, 1), ylims = extrema(R)))
    sleep(0.005)
end

for n = 100:200
    display(sticks(x, R[n, :]; xlims = (0, 1), ylims = extrema(R)))
    sleep(0.005)
end

surface(R; xflip = true)
surface(W; xflip = true)
surface(W)
plotmat(W)

c = @. M / N - σ^2
c[1] = 0

T = sqrt(M / N) * Ξ[:, 1:M]' + Diagonal(sqrt.(c)) * Ψ'
plotmat(T)
plotmat(Diagonal(sqrt.(c)) * Ψ')

plotmat(I / N - W'W / M)
plot!(y, ke.(P * V * u); label = "PVu")
# plot!(x, ke.(W * u);  label = "Wu")

plot(; xlabel = "x", title = "Kinetic energy")
plot!(y, u; label = "u")
plot!(y, R * W * u; label = "RWu")
plot!(y, P * V * u; label = "PVu")

plot(; xlabel = "x", title = "Kinetic energy")
plot!(y, u; label = "u")
plot!(x, W * u; label = "Wu")
plot!(y, P * V * u; label = "PVu")
plot!(y[M+1:end], V * u; label = "Vu")

plot(; yscale = :log10, xlabel = "x", title = "Kinetic energy")
# scatter!(y, abs.(P * V * u); label = "PVu")
# scatter!(y[M+1:end], abs.(V * u); label = "Vu")

norm(u) / N
norm(W * u) / M + norm(V * u) / (N - M)

for c ∈ collect(eachcol(nullspace(Matrix(W))))[100:200]
    display(plot(y, c))
end

sum(ke.(W * u)) / M
sum(ke.(R * W * u)) / N
sum(ke.(P * V * u)) / N
sum(ke.(R * W * u)) / N + sum(ke.(P * V * u)) / N
sum(ke, u) / N

plot(; xlabel = "SVD index", yscale = :log10)
scatter!(σ; label = "σ")
scatter!(abs.(Σ * Ψ' * u); label = "|ΣΨ'u|")
scatter!(abs.(Ψ' * u); label = "|Ψ'u|")

σ

e = zeros(N)
e[250] = 1

plot(; xlabel = "x", title = "Signal")
plot!(y, e; label = "e")
# plot!(x, W * e;  label = "RWe")
plot!(y, R * W * e; label = "RWe")
# plot!(y, e - R * W * e; label = "RWe")

plot(y, abs.(R * W * e); yticks = 10.0 .^ (-8:0), yscale = :log10, label = "RWe")

plot(x, abs.(R[500, :]); yscale = :log10)
plot(y, abs.(R[:, 31]); yscale = :log10)
