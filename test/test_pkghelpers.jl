using Test
using SchottkyAnoMaLy

@info "Testing PackageHelpers" 
@time @testset "Testing PackageHelpers" begin
    
    @testset "Throws Errors" begin

        @testset "AssertionErrors" begin
            # Test chebycoefficients
            all_ensembles = rand(RandomDonutVolcanoGenerator(50, 10, 10))
            num_ens = length(all_ensembles)
            order = 22
            coeffs = zeros( num_ens == 2 ? 3 : 2, order + 1 )
            @test_throws AssertionError chebycoefficients!(coeffs, all_ensembles, order, 0, 100)
            
            coeffs = zeros( num_ens, order )
            @test_throws AssertionError chebycoefficients!(coeffs, all_ensembles, order, 0, 100)
        end

        @testset "MethodErrors" begin
            # Test create_chebinterp
            @test_throws MethodError create_chebinterp(rand(Float64, 3, 4), 0., 1.)
        end
        
    end

end