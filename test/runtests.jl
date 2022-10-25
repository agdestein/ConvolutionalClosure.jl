using Aqua
using ConvolutionalClosure
using Test

@testset "ConvolutionalClosure.jl" begin
    # Write your tests here.
    @testset "Aqua" begin
        Aqua.test_all(
            ConvolutionalClosure;
            ambiguities = false,
            project_toml_formatting = false, # https://github.com/JuliaTesting/Aqua.jl/issues/72
        )
    end
end
