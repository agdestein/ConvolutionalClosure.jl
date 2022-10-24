using Literate

examples = ["main.jl"]

for e ∈ examples
    Literate.markdown(e, ".")
    Literate.notebook(e, "."; execute = false)
end
