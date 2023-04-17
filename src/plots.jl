"""
    plotsol(x, t, u; kwargs...)

Plot solution u.
"""
plotsol(x, t, u; kwargs...) = heatmap(
    t,
    x,
    u;
    xlabel = "t",
    ylabel = "x",
    # aspect_ratio = :equal,
    kwargs...,
)

"""
    plotfield(x, y, u; kwargs...)

Plot solution u.
"""
plotfield(x, y, u; kwargs...) = heatmap(
    x,
    y,
    u';
    xlabel = "x",
    ylabel = "y",
    # aspect_ratio = :equal,
    kwargs...,
)

"""
    plotmat(A; kwargs...)

Plot matrix.
"""
plotmat(A; kwargs...) = heatmap(
    A;
    # reverse(A; dims = 1);
    # aspect_ratio = :equal,
    xlims = (1 / 2, size(A, 2) + 1 / 2),
    ylims = (1 / 2, size(A, 1) + 1 / 2),
    yflip = true,
    xmirror = true,
    # xticks = nothing,
    # yticks = nothing,
    kwargs...,
)


# heatmap(reverse(A'); dims = 2))

plotmat(A::AbstractSparseMatrix; kwargs...) = plotmat(Matrix(A); kwargs...)
