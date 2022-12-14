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
                Ndim = rand(2:5)   # probably overkill
                Nsize = Int(2^3)
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
            μdist = Uniform(0, 50)
            σdist = Uniform(0.001, 50)
            for idx ∈ UnitRange(1, 10)
                μ, σ = rand(μdist), rand(σdist)
                fullinteg = quadgk( x -> donutvolcano(x, μ, σ), -Inf, Inf )
                integ = quadgk( x -> donutvolcano(x, μ, σ), 0., Inf )
                if integ[1] ≈ 0. || fullinteg[1] ≈ 0.
                    @show μ, σ
                end
                @test abs(integ[1] - oneunit(μ)) / oneunit(μ) ≤ integ[2]
                @test abs(fullinteg[1] - oneunit(μ)) / oneunit(μ) ≤ fullinteg[2]
            end
        end

    end

    @testset "Test DonutVolcanoEnsemble" begin
        
        @testset "Constructors" begin
            
            @testset "Assertion errors" begin
                @test_throws AssertionError DonutVolcanoEnsemble{Float64}([(-1.0, 1.0)])
                @test_throws AssertionError DonutVolcanoEnsemble{Float64}([(1.0, -1.0)])
                
                @test_throws AssertionError DonutVolcanoEnsemble([(-1.0, 1.0)])
                @test_throws AssertionError DonutVolcanoEnsemble([(1.0, -1.0)])
            end

            @testset "Type conversions" begin
                # Type 1 Constructor
                dve = DonutVolcanoEnsemble( [(0., 1.)] )
                @test eltype(dve) === Float64
                # Type 2 Constructor
                dve = DonutVolcanoEnsemble( [(0., 1)] )
                @test eltype(dve) === Float64
                # Type 3 Constructor
                dve = DonutVolcanoEnsemble(Float32, [(0, 1)] )
                @test eltype(dve) === Float32
                # Type 4 Constructor
                dve = DonutVolcanoEnsemble(Float16)
                @test eltype(dve) === Float16
                # Type 5 Constructor
                dve = DonutVolcanoEnsemble()
                @test eltype(dve) === Float64
            end

            @testset "Empty constructors" begin
                dve = DonutVolcanoEnsemble()
                @test length(dve) == length(ensemble(dve)) == zero(Int)
                
                dve = DonutVolcanoEnsemble(Int32)
                @test length(dve) == length(ensemble(dve)) == zero(Int)
            end

            @testset "Push and Append protection" begin
                dve = DonutVolcanoEnsemble()
                @test_throws AssertionError push!(dve, (-1, 1))
                @test_throws AssertionError push!(dve, (1, -1))
                @test_throws AssertionError push!(dve, (1, 0))
                
                @test_throws AssertionError append!(dve, [(-1, 1)])
                @test_throws AssertionError append!(dve, [(1, -1)])
                @test_throws AssertionError append!(dve, [(1, 0)])
            end

        end

    end

    @testset "Test RandomDonutVolcanoGenerator" begin
        
        @testset "Assertion errors" begin        
            @test_throws AssertionError RandomDonutVolcanoGenerator(3, 10, 10, 0, 0., 1e-2)
            @test_throws AssertionError RandomDonutVolcanoGenerator(0, 10, 10, 1, 0., 1e-2)
            
            @test_throws AssertionError RandomDonutVolcanoGenerator(3, 10, 10, 1, -1., 1e-2)
            @test_throws AssertionError RandomDonutVolcanoGenerator(3, -1, 10, 1, 0., 1e-2)
            
            @test_throws AssertionError RandomDonutVolcanoGenerator(3, 10, 10, 1, 0., 0.)
            @test_throws AssertionError RandomDonutVolcanoGenerator(3, 10, 10, 1, 0., -1.0)
            @test_throws AssertionError RandomDonutVolcanoGenerator(3, 10, -1, 1, 0., 1e-2)
        end

        @testset "Random ensembles" begin
            rdveg = RandomDonutVolcanoGenerator(3, 10, 10)
            num_ens = 100
            @test length( rand(rdveg, num_ens) ) == num_ens
        end

    end

end
println()