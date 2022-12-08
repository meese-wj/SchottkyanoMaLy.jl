using Test
using SchottkyAnoMaLy
using Documenter

@info "Testing documentation"
@time @testset "DocTests" begin
    DocMeta.setdocmeta!(SchottkyAnoMaLy, :DocTestSetup, :(using SchottkyAnoMaLy); recursive=true)
    doctest(SchottkyAnoMaLy)
end
