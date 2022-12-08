using Test
using SchottkyAnoMaLy
using QuadGK
using Distributions

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

    @testset "Test DonutVolcano distribution" begin
        
        @testset "Negative values vanish" begin
            vals = -10:-1
            donuts = donutvolcano(vals, 0., 1.)
            @test all(donuts .== zero.(vals))
        end
        
        @testset "Nonnegative domain contributes" begin
            vals = -10:10
            donuts = donutvolcano(vals, 0., 1.)
            @test length(donuts[donuts .== zero(eltype(donuts))]) == length(collect(vals)) ÷ 2 + 1  # donutvolcano(0) == 0 !
            @test length(donuts[donuts .> zero(eltype(donuts))]) == length(collect(vals)) ÷ 2
        end
        
        @testset "Nonnegativity of the range" begin
            vals = LinRange(-10, 10, 1_000_000)
            donuts = donutvolcano(vals, 0., 1.)
            @test all(donuts .>= zero(eltype(donuts)))  # donutvolcano(0) == 0 !
        end

        @testset "Normalization" begin
            μdist = Uniform(0, 1000)
            σdist = Uniform(0.001, 1000)
            for idx ∈ UnitRange(1, 10)
                μ, σ = rand(μdist), rand(σdist)
                fullinteg = quadgk( x -> donutvolcano(x, μ, σ), -Inf, Inf )
                integ = quadgk( x -> donutvolcano(x, μ, σ), 0., Inf )
                @test abs(integ[1] - oneunit(μ)) / oneunit(μ) ≤ integ[2]
                @test abs(fullinteg[1] - oneunit(μ)) / oneunit(μ) ≤ fullinteg[2]
            end
        end

    end


end
println()