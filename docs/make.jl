using ConvolutionalClosure
using Documenter

DocMeta.setdocmeta!(ConvolutionalClosure, :DocTestSetup, :(using ConvolutionalClosure); recursive=true)

makedocs(;
    modules=[ConvolutionalClosure],
    authors="Syver Døving Agdestein <syverda@gmail.com> and contributors",
    repo="https://github.com/agdestein/ConvolutionalClosure.jl/blob/{commit}{path}#{line}",
    sitename="ConvolutionalClosure.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://agdestein.github.io/ConvolutionalClosure.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/agdestein/ConvolutionalClosure.jl",
    devbranch="main",
)
