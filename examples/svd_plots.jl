if isdefined(@__MODULE__, :LanguageServer)
    include("svd.jl")
end

plotmat(W; title = "W")

plotmat(W .!= 0; title = "Discrete filter (sparsity)")

plot(W[10, :]; yscale = :log10, ylims = (1e-8, 1))
plot(W[10, :])

plot(; xlabel = "x", title = "Local filter kernels")
sticks!(y, W[10, :]; label = @sprintf "x = %.2f" x[10])
sticks!(y, W[20, :]; label = @sprintf "x = %.2f" x[20])
sticks!(y, W[30, :]; label = @sprintf "x = %.2f" x[30])
sticks!(y, W[40, :]; label = @sprintf "x = %.2f" x[40])
sticks!(y, W[50, :]; label = @sprintf "x = %.2f" x[50])

# savefig(loc * "local_filter_kernels.pdf")

plotmat(R)

plotmat(W * W')
plotmat(W' * W)
plotmat(R * W)
plotmat(W * R)

plotmat(R'R / N)

plotmat(Φ * Ψ')

# k = 9
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
# indind = k -> 1:k
indind = k -> k:k
plot(
    (
        plotmat(
            Φ[:, indind(k)] * Diagonal(σ[indind(k)]) * Ψ[:, indind(k)]';
            xticks = iplot ≤ 4,
            yticks = iplot ∈ [1, 5, 9],
            # xlabel = "x",
            colorbar = false,
            title = iplot == 1 ? "k = $k" : "$k",
        ) for (iplot, k) in enumerate(i)
    )...;
    # layout = (2, 1),
    # title = "Eigenvectors of I/N - W'W/M",
    # size = (1200, 800),
    size = (800, 500),
    # plot_title = "Truncated filter matrix",
)

# savefig(loc * "Wk.pdf")
# savefig(loc * "Wkall.pdf")

plotmat(V)
plotmat(P)
plotmat(P * V)
plotmat(Ψ)
plotmat(Φ)
plotmat(Φ * Ψ')
plotmat(Ψ * Ψ')

i = 200
plot(; title = "i = $i", xlabel = "x")
plot!(y, √N * Ψ * Ψ[i, :]; label = "√N ΨΨ'ᵢ")
plot!(x, √M * Φ * Ψ[i, :]; label = "√M ΦΨ'ᵢ")

i = 3
plot(; title = "i = $i", xlabel = "x")
plot!(y, √N * Ψ[:, i]; label = "√N ψ")
plot!(x, √M * Φ[:, i]; label = "√M ϕ")

i = 2;
plot(y, Ψ[:, i]; title = "Right singular vectors $i", xlabel = "x");
i = 316;
plot(y, P[:, i]; title = "Right singular vectors $i", xlabel = "x");

for i = 1:M
    display(plot(y, Ψ[:, i]; title = "Right singular vectors $i", xlabel = "x"))
    sleep(0.1)
end

for k = 300:400
    pl = plot(
        y,
        P[:, k];
        legend = false,
        title = "ξₖ, k = $k",
        xlabel = "x",
        ylims = (-0.2, 1.0),
    )
    vspan!(pl, [0.0, M / N]; fillalpha = 0.1, color = 1)
    display(pl)
    sleep(0.05)
end

i = [1:7; 10; 20; 30; 40; 50]
plot(
    (
        begin
            plot(;
                # label = i',
                xticks = iplot ≥ 9,
                xlabel = iplot ≥ 9 ? "x" : "",
                yticks = false,
                # xlabel = "x",
                title = iplot == 1 ? "k = $k" : "$k",
                # title = m,
                legend = iplot == 1,
            )
            plot!(y, √N * Ψ[:, k]; label = "√N ψₖ")
            plot!(x, √M * Φ[:, k]; label = "√M ϕₖ")
        end for (iplot, k) in enumerate(i)
    )...;
    # layout = (2, 1),
    # plot_title = "Singular vectors ψₖ and ϕₖ",
    # plot_title = "Singular vectors",
    # size = (1200, 800),
    size = (800, 500),
)

# savefig(loc * "singular_vectors.pdf")

i = [1, 90, 180, 270, 360, 450]
plot(
    (
        begin
            plot(;
                legend = false,
                xticks = iplot ≥ 4,
                xlabel = iplot ≥ 4 ? "x" : "",
                yticks = false,
                # xlabel = "x",
                title = "k = $i",
                xlims = (0, l()),
            )
            plot!(y, Ξ[:, i])
            vspan!([0.0, M / N]; fillalpha = 0.1, color = 1, label = false)
        end for (iplot, i) in enumerate(i)
    )...;
    # layout = (2, 1),
    # size = (1200, 800),
    # plot_title = "Right singular vectors ξₖ",
    size = (800, 500),
)

# savefig(loc * "singular_vectors_zero.pdf")

σ

plot(; xlabel = "k", title = "Singular values")
scatter!(σ; label = "σₖ")
hline!([sqrt(M / N)]; label = "σ₁ = √(M/N)")

# savefig(loc * "singular_values.pdf")

Ψ' * Ψ

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
plot!(y, ke.(P * V * u); label = "PVu")
plot!(x, ke.(W * u); label = "Wu")

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
# sticks(x, R[s*m, :]; xlims = (0, 1))

for m in [1:M; 1:M]
    display(plot(y, R[:, m]; xlims = (0, 1), ylims = extrema(R)))
    sleep(0.005)
end

for n = 100:10:500
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

X = sqrt(s) * Ψ * Σ * Ψ'

plotmat(X)

W2 = 1 / √s * Φ * Ψ'
R2 = √s * Ψ * Φ'

plotmat(W2)
plotmat(R2)
plotmat(W2 * R2)
plotmat(R)
plotmat(R2 * W)
plotmat(R * W)

PV = Ξ * Ξ' + Ψ * (I - √s * Σ) * Ψ'

dec = eigen(PV)

scatter(abs.(dec.values))

plotmat(PV)

i = 3
plot(y, R2[:, i])
plot!(y, R[:, i])

plot(; xlabel = "x", title = "Signal")
plot!(y, u; label = "u")
plot!(y, R * W * u; label = "RWu")
plot!(y, X * u; label = "Xu")
plot!(x, W * u; label = "Wu")
plot!(y, P * V * u; label = "PVu")
plot!(y, u - X * u; label = "u - Xu")

plot(; xlabel = "x", title = "Kinetic energy")
plot!(y, u .^ 2; label = "u")
# plot!(x, (W * u).^2; label = "Wu")
plot!(y, (R * W * u) .^ 2; label = "RWu")
plot!(y, (R * W * u) .^ 2 .+ (P * V * u) .^ 2; label = "RWu + PVu")
# plot!(y, (P * V * u) .^ 2; label = "PVu")
# plot!(y[M+1:end], V * u; label = "Vu")

plot(; xlabel = "x")
plot!(y, u; label = "u")
# savefig(loc * "u1.pdf")
current()

plot(; xlabel = "x")
plot!(y, u; label = "u")
plot!(x, Φ * Σ * Ψ' * u; label = "ΦΣΨ'u");
# savefig(loc * "u2.pdf")
current()

plot(; xlabel = "x")
plot!(y, u; label = "u")
plot!(x, Φ * Σ * Ψ' * u; label = "ΦΣΨ'u");
plot!(y, Ψ * Ψ' * u; label = "ΨΨ'u")
# savefig(loc * "u3.pdf")
current()

plot(; xlabel = "x")
plot!(y, u; label = "u")
plot!(x, Φ * Σ * Ψ' * u; label = "ΦΣΨ'u");
plot!(y, √s * Ψ * Σ * Ψ' * u; label = "√sΨΣΨ'u")
# savefig(loc * "u4.pdf")
current()

plot(; xlabel = "x")
plot!(y, u; label = "u")
plot!(x, 1 / √s * Φ * Ψ' * u; label = "1/√sΦΨ'u");
plot!(y, Ψ * Ψ' * u; label = "ΨΨ'u")
# savefig(loc * "u5.pdf")
current()

plot(; xlabel = "x", title = "Pseudo-inverse reconstructor")
plot!(y, u; label = "u")
plot!(y, Ψ * Ψ' * u; label = "RWu")
plot!(y, (I - Ψ * Ψ') * u; label = "PVu")
# savefig(loc * "example_signal.pdf")
current()

plot(; yscale = :log10, xlabel = "x", title = "Kinetic energy")
scatter!(y, abs.(P * V * u); label = "PVu")
scatter!(y[M+1:end], abs.(V * u); label = "Vu")

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

plotmat(R * W)

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
