using Test
using SchottkyAnoMaLy

@info "Testing DonutVolcanoEnsembles"
@time @testset "Testing DonutVolcanoEnsembles" begin
    
    @testset "Test Theta function" begin
        
        @testset "Scalar values" begin
            # generate a random number on [0, 1]
            temp = rand()
            @test SchottkyAnoMaLy.Theta( -temp ) == zero(temp)         # -temp < 0
            @test SchottkyAnoMaLy.Theta( temp )  == oneunit(temp)      # temp > 0
            @test SchottkyAnoMaLy.Theta( zero(temp) ) == oneunit(temp) # Theta(0) ≡ 0 for us
        end
        
        let
            @testset "Vector values" begin
                # generate a random numbers on [0, 1]
                Nsize = Int(2^20)
                temps = rand(Nsize)
                @test all( SchottkyAnoMaLy.Theta( -1 .* temps ) .== zero.(temps) )         # -temp < 0
                @test all( SchottkyAnoMaLy.Theta( temps ) .== oneunit.(temps) )            # temp > 0
                @test all( SchottkyAnoMaLy.Theta( zero.(temps) ) .== oneunit.(temps) )     # Theta(0) ≡ 0 for us
            end
        end
        
        let
            @testset "Array values" begin
                # generate a random numbers on [0, 1]
                Ndim = rand(2:10)   # probably overkill
                Nsize = Int(2^5)
                dims = ntuple(x -> Nsize, Ndim)
                temps = rand(Float64, dims)
                @test all( SchottkyAnoMaLy.Theta( -1 .* temps ) .== zero.(temps) )         # -temp < 0
                @test all( SchottkyAnoMaLy.Theta( temps ) .== oneunit.(temps) )            # temp > 0
                @test all( SchottkyAnoMaLy.Theta( zero.(temps) ) .== oneunit.(temps) )     # Theta(0) ≡ 0 for us
            end
        end

    end

    


end
println()