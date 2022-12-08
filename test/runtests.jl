using Pkg
Pkg.activate(@__DIR__)

using SchottkyAnoMaLy
using Test

@testset "SchottkyAnoMaLy.jl" begin

    include("test_DVE.jl")
    include("doctests.jl")

end
