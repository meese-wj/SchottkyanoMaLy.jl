using SchottkyAnoMaLy
using Test

@testset "SchottkyAnoMaLy.jl" begin

    include("test_statmech.jl")
    include("test_DVE.jl")
    include("test_cVLA.jl")
    # include("doctests.jl")

end
