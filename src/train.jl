function train(
    loss,
    p₀,
    niter;
    opt = Optimisers.ADAM(0.001),
    callback = (i, p) -> println("Iteration $i"),
    ncallback = 1,
)
    p = p₀
    opt = Optimisers.setup(opt, p)
    callback(0, p)
    for i ∈ 1:niter
        grad = first(gradient(loss, p))
        opt, p = Optimisers.update(opt, p, grad)
        # i % ncallback == 0 ? callback(i, p) : println("Iteration $i")
        i % ncallback == 0 && callback(i, p)
    end
    p
end
