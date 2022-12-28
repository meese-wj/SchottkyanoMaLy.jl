using Test
using SchottkyAnoMaLy

@info "Testing SpecificHeatLinearAlgebra" 
@time @testset "Testing SpecificHeatLinearAlgebra" begin
    
    @testset "AssertionErrors" begin
        # Test populate_msqdiffs
        cV_arr = zeros(4, 5)
        temps = zeros(3)
        @test_throws AssertionError populate_msqdiffs(cV_arr, temps)

        temps = zeros(4)
        mat = zeros(6, 6)
        @test_throws AssertionError populate_msqdiffs!(mat, cV_arr, temps)

        # Test input_reference_msqdiffs
        temps = zeros(10)
        cV_arr = zeros(9, 9)
        test_cV = zeros(9)
        @test_throws AssertionError input_reference_msqdiffs(test_cV, cV_arr, temps)

        cV_arr = zeros(10, 9)
        @test_throws AssertionError input_reference_msqdiffs(test_cV, cV_arr, temps)
        
        output = zeros(9)
        @test_throws AssertionError input_reference_msqdiffs!(output, test_cV, cV_arr, temps)
        
    end

end