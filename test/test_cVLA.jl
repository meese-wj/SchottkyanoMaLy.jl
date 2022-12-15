using Test
using SchottkyAnoMaLy

@info "Testing SpecificHeatLinearAlgebra" 
@time @testset "Testing SpecificHeatLinearAlgebra" begin
    
    @testset "AssertionErrors" begin
        cV_arr = zeros(4, 5)
        temps = zeros(3)
        @test_throws AssertionError populate_msqdiffs(cV_arr, temps)

        temps = zeros(4)
        mat = zeros(6, 6)
        @test_throws AssertionError populate_msqdiffs!(mat, cV_arr, temps)
    end

end