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

        all_ensembles = rand(RandomDonutVolcanoGenerator(50, 10, 10))
        num_ens = length(all_ensembles)
        order = 22
        coeffs = zeros( num_ens == 2 ? 3 : 2, order + 1 )
        @test_throws AssertionError chebycoefficients!(coeffs, all_ensembles, order, 0, 100)
        
        coeffs = zeros( num_ens, order )
        @test_throws AssertionError chebycoefficients!(coeffs, all_ensembles, order, 0, 100)
        
    end

end