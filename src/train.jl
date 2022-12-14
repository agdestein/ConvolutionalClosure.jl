function train(loss, pā, niter; opt = Optimisers.ADAM(0.001), callback = (i, p) -> println("Iteration $i"), ncallback = 1)
    p = pā
    opt = Optimisers.setup(opt, p)
    callback(0, p)
    for i ā 1:niter
        grad = first(gradient(loss, p))
        opt, p = Optimisers.update(opt, p, grad)
        # i % ncallback == 0 ? callback(i, p) : println("Iteration $i")
        i % ncallback == 0 && callback(i, p)
    end
    p
end
