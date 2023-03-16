if isdefined(@__MODULE__, :LanguageServer) #src
    include("svd.jl")                      #src
end                                        #src

using LaTeXStrings

anim = @animate for k = 1:M
    # for k = 1:M
    ind = 1:k
    Wk = Φ[:, ind] * Diagonal(σ[ind]) * Ψ[:, ind]'
    plWk = plotmat(Wk; title = "∑ᵢᵏ σᵢ ϕᵢ ψᵢᵀ", colorbar = false)
    Wi = Φ[:, k] * Ψ[:, k]'
    plWi = plotmat(Wi; title = "ϕₖ ψₖᵀ", colorbar = false)
    plψk = plot(y, Ψ[:, k]; yticks = false, title = "ψₖ", xlabel = "x", legend = false)
    plσ = scatter(
        σ[ind];
        xlims = (0, M),
        ylims = (0, 1.05 * σ[1]),
        xlabel = "i",
        title = "σᵢ",
        color = [fill(1, k - 1); 2],
        legend = false,
    )
    pl = plot(plWi, plWk, plψk, plσ; plot_title = "k = $k", size = (800, 500))
    # display(pl)
    # sleep(0.1)
end

gif(anim, loc * "animations/svd.gif"; fps = 10)
