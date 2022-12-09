using Test
using SchottkyAnoMaLy
using BenchmarkTools

@info "Testing StatisticalMechanicsFormulas"
@time @testset "Stat Mech Tests" begin
    
    @testset "Specific Heat" begin
        @testset "Method Errors" begin
            struct doomed_to_fail <: SchottkyAnoMaLy.NLevelSystem end
            @test_throws MethodError specific_heat(doomed_to_fail, 1, 1) # No method implemented for the doomed_to_fail subtype
        end

        @testset "Benchmarking" begin
            T, Δ = 1, 1
            bm = @benchmark specific_heat($T, $Δ)
            @test bm.allocs == zero(bm.allocs)
            
            bm = @benchmark specific_heat(TwoLevelSystem(), $T, $Δ)
            @test bm.allocs == zero(bm.allocs)
            
            Tvals = collect(1:10)
            bm = @benchmark specific_heat($Tvals, $Δ) 
            @test bm.allocs == oneunit(bm.allocs)  # One allocation for the broadcast

            bm = @benchmark specific_heat(TwoLevelSystem(), $Tvals, $Δ) 
            @test bm.allocs == oneunit(bm.allocs)  # One allocation for the broadcast
        end

        @testset "Type Stability" begin
            Tvals = collect(1:10)
            @test_nowarn @inferred specific_heat(Tvals[1], 1)
            @test_nowarn @inferred specific_heat(Tvals, 1)
            
            @test_nowarn @inferred specific_heat(TwoLevelSystem(), Tvals[1], 1)
            @test_nowarn @inferred specific_heat(TwoLevelSystem(), Tvals, 1)
        end
    end


end
println()