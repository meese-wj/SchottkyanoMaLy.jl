using Random
using Distributions
using SpecialFunctions
using QuadGK
import Base: eltype, push!, append!, rand

# Base overloads
export eltype, push!, append!, rand
# DonutVolcanoEnsemble stuff
export donutvolcano, DonutVolcanoEnsemble, ensemble, μvalue, σvalue,
       specific_heat, RandomDonutVolcanoGenerator

@doc raw"""
    Theta(x)

Package-specific implementation of the Heaviside Theta function. Our definition is

```math
\Theta(x) = \begin{cases}
1, & x \geq 0
\\
0, & \mathrm{otherwise}
\end{cases}.
```
"""
Theta(x) = map( val -> convert(eltype(val), val ≥ zero(val)), x )

@doc raw"""
    donutvolcano(x, μ, σ)

The `donutvolcano` distribution calculated for an input `x` with center `μ` and width `σ`.
The argument `x` can be a scalar or broadcastable.

The distribution itself is given by 

```math
{\rm donutvolcano}(x; μ, σ) = \frac{x \Theta(x)}{\mathcal{N}(\mu,\sigma)} \exp\left[ -\frac{\left(x - \mu\right)^2}{2\sigma^2} \right],
```

where we use our own Heaviside [`Theta`](@ref) function above. The normalization constant ``\mathcal{N}(\mu,\sigma)`` given by [`norm_donutvolcano`](@ref).
"""
function donutvolcano(x, μ, σ)
	normalizer = norm_donutvolcano(μ, σ)
	return @. x * Theta(x) * exp( -0.5 * ( (x - μ) / σ )^2 ) / normalizer
end

@doc raw"""
    norm_donutvolcano(μ, σ)

The normalization for a [`donutvolcano`](@ref) distribution with center `μ` and width `σ`.
The explicit form is given by

```math
\mathcal{N}(\mu, \sigma) = \sqrt{\frac{\pi }{2}} \mu  \sigma 
   \left[\text{erf}\left(\frac{\mu }{\sqrt{2} \sigma
}\right)+1\right]+\sigma ^2 {\rm e}^{-\frac{\mu ^2}{2 \sigma ^2}}.
```
"""
function norm_donutvolcano(μ, σ)
	arg = μ / (σ * sqrt(2))
	return σ^2 * exp( -arg^2 ) + sqrt(π/2) * μ * σ * (oneunit(μ) + erf(arg))
end

@doc raw"""
    DonutVolcanoEnsemble{T <: Real}

An ensemble of ``M`` [`donutvolcano`](@ref)s takes the form 

```math
\mathrm{p}_M(x) = \frac{1}{M} \sum_{k=1}^M \mathrm{donutvolcano}(x, \mu_k, \sigma_k).
```

This object is a wrapper around an `ensemble::Vector{Tuple{T, T}}` for the individual
[`donutvolcano`](@ref)s. The values of ``(μ_k, σ_k)`` are restricted such that ``\mu_k \geq 0``
and ``\sigma_k > 0`` via the [`valid`](@ref) function.
"""
struct DonutVolcanoEnsemble{T <: Real}
	ensemble::Vector{Tuple{T, T}}

    DonutVolcanoEnsemble{T}( tups::Vector{Tuple{T, T}} ) where T = ( map(valid, tups); new{T}(tups) )
end

"""
    valid(::Tuple{T, T})

Returns the given `Tuple` `(μ, σ)` if `μ ≥ 0` and `σ > 0`. Otherwise, an `AssertionError` 
will be thrown.
"""
function valid(tup)
    # TODO: Sweep μ/σ ratio for when quadgk fails. This will give bounds on σ in terms of μ. 
    # (μ/σ ≈ 32/0.17 ≈ 188 is problematic from tests) 
    @assert μvalue(tup) ≥ zero(μvalue(tup)) "All μ values must be strictly nonnegative."
    @assert σvalue(tup) > zero(σvalue(tup)) "All σ values must be strictly positive."
    return tup
end

# Constructors
"""
    DonutVolcanoEnsemble(args...)

Various interfaces to a [`DonutVolcanoEnsemble`](@ref)

# Copy/Move constructors

* `DonutVolcanoEnsemble(::Vector{Tuple{T, T}})`
    * Create an ensemble from a `Vector` of `(μ, σ)` `Tuple`s of the same `Type`.

* `DonutVolcanoEnsemble(::Vector{Tuple{T, U}})`
    * Create an ensemble from a `Vector` of `(μ, σ)` `Tuple`s of the different `Type`s. 
    * The type of the ensemble is chosen by `Base.promote_type`.

* `DonutVolcanoEnsemble(::Type{S}, ::Vector{Tuple{T, U}})`
    * Create an ensemble from a `Vector` of `(μ, σ)` `Tuple`s of the different `Type`s into an ensemble of with preferred type `S`.
    * The ultimate type of the ensemble is chosen by `Base.promote_type`.

!!! note
    The constructors above will check that any supplied `pair` is [`valid`](@ref).
    
# Empty constructors

* `DonutVolcanoEnsemble(::Type{S})`
    * Create an empty ensemble of chosen type `S`.

* `DonutVolcanoEnsemble()`
    * Create an empty ensemble. By default the type is `Float64`.
"""
DonutVolcanoEnsemble( tups::Vector{Tuple{T, T}} ) where T = DonutVolcanoEnsemble{T}( tups )	
function DonutVolcanoEnsemble(tups::Vector{Tuple{T, U}}) where {T, U}
	t = promote_type(T, U)
	new_tups = map( tup -> convert.(t, tup), tups )
	return DonutVolcanoEnsemble(new_tups)
end
function DonutVolcanoEnsemble(::Type{S}, tups::Vector{Tuple{T, U}}) where {S, T, U}
	dve = DonutVolcanoEnsemble(tups)
	t = promote_type( eltype(dve), S )
	return DonutVolcanoEnsemble( map( tup -> convert.(t, tup), ensemble(dve) ) )
end
## Empty constructors
DonutVolcanoEnsemble(::Type{T}) where T = DonutVolcanoEnsemble{T}( Tuple{T, T}[] )
DonutVolcanoEnsemble() = DonutVolcanoEnsemble(Float64)

# Getters
"""
    ensemble(::DonutVolcanoEnsemble)

Getter for the `Vector{Tuple{T, T}}` of the given ensemble.
"""
ensemble(dve::DonutVolcanoEnsemble) = dve.ensemble
"""
    get_pair(::DonutVolcanoEnsemble, pair_idx) -> Tuple{T, T}

Getter for a single pair (`Tuple{T, T}`) at a specified `pair_idx`.
"""
get_pair(dve::DonutVolcanoEnsemble, pair_idx) = Tuple( ensemble(dve)[pair_idx] )
"""
    npairs(::DonutVolcanoEnsemble)

Return the number of [`donutvolcano`](@ref) in the [`DonutVolcanoEnsemble`](@ref).
"""
npairs(dve::DonutVolcanoEnsemble) = length(ensemble(dve))
"""
    μvalue(::Tuple{T, T})
    μvalue(::DonutVolcanoEnsemble, idx)

Return the value of μ from a given `(μ, σ)` `Tuple`.
"""
μvalue(tup::Tuple{T, T}) where T = tup[1]
μvalue(dve::DonutVolcanoEnsemble, idx) = μvalue(get_pair(dve, idx))
"""
    σvalue(::Tuple{T, T})
    σvalue(::DonutVolcanoEnsemble, idx)

Return the value of σ from a given `(μ, σ)` `Tuple`.
"""
σvalue(tup::Tuple{T, T}) where T = tup[2]
σvalue(dve::DonutVolcanoEnsemble, idx) = σvalue(get_pair(dve, idx))

# Convenient Base overloads
"""
    Base.eltype(::DonutVolcanoEnsemble)

Return the type of the [`DonutVolcanoEnsemble`](@ref).

```jldoctest
julia> dve = DonutVolcanoEnsemble()
DonutVolcanoEnsemble{Float64}(Tuple{Float64, Float64}[])

julia> eltype(dve)
Float64
```
"""
Base.eltype(::DonutVolcanoEnsemble{T}) where T = T
"""
    Base.push!(::DonutVolcanoEnsemble, pair)

Push a new `(μ, σ)` `Tuple` into a given [`DonutVolcanoEnsemble`](@ref).

```jldoctest
julia> dve = DonutVolcanoEnsemble()
DonutVolcanoEnsemble{Float64}(Tuple{Float64, Float64}[])

julia> push!(dve, (0, 1))
1-element Vector{Tuple{Float64, Float64}}:
 (0.0, 1.0)

julia> dve
DonutVolcanoEnsemble{Float64}([(0.0, 1.0)])
```

!!! note
    This function will check that the `pair` is [`valid`](@ref).
"""
Base.push!(dve::DonutVolcanoEnsemble{T}, pair) where T = push!(ensemble(dve), convert.(T, pair) |> valid )
"""
    Base.append!(::DonutVolcanoEnsemble, vals)

Append the `vals::Vector{Tuple{S, T}}` to the [`DonutVolcanoEnsemble`](@ref).

```jldoctest
julia> dve = DonutVolcanoEnsemble()
DonutVolcanoEnsemble{Float64}(Tuple{Float64, Float64}[])

julia> append!(dve, [(0, 1), (1, 3)])
2-element Vector{Tuple{Float64, Float64}}:
 (0.0, 1.0)
 (1.0, 3.0)

julia> dve
DonutVolcanoEnsemble{Float64}([(0.0, 1.0), (1.0, 3.0)])
```

!!! note
    This function will check that the `pair` is [`valid`](@ref).
"""
Base.append!(dve::DonutVolcanoEnsemble{T}, vals) where T = append!(ensemble(dve), map(x -> convert.(T, x) |> valid, vals))

"""
    (::DonutVolcanoEnsemble)(x)

Evaluate the DonutVolcanoEnsemble as a function over a set of values `x`.

```jldoctest
julia> dve = DonutVolcanoEnsemble(Float64, [(0, 1)]);

julia> dve(2)
0.2706705664732254

julia> dve([3, 4])
2-element Vector{Float64}:
 0.033326989614726917
 0.0013418505116100474
```
"""
function (dve::DonutVolcanoEnsemble)(x)
	output = zero.(x)
	@inbounds for pdx ∈ UnitRange(1, npairs(dve))
		output += donutvolcano(x, get_pair(dve, pdx)...)
	end
	return output ./ npairs(dve)
end

@doc raw"""
    specific_heat([::NLevelSystem = TwoLevelSystem()], T, ::DonutVolcanoEnsemble, [Δmin = 0, Δmax = Inf])

Calculate the [`specific_heat`](@ref) ``\tilde{c}_V(T, \Delta`` integrated over the [`DonutVolcanoEnsemble`](@ref)
for a fixed temperature ``T``. By default, this is calculated for a [`TwoLevelSystem`](@ref).
    
The formula for this integration is given by 

```math
c_V(T) = \int_{\Delta_\min}^{\Delta_\max} \mathrm{d}\Delta\, \mathrm{p}(\Delta) \tilde{c}_V(T, \Delta),
```

where ``\mathrm{p}(\Delta)`` is calculated from the [`DonutVolcanoEnsemble`](@ref).
"""
function specific_heat(nls::NLevelSystem, T, dve::DonutVolcanoEnsemble, Δmin = 0, Δmax = Inf)
    return quadgk( Δ -> dve(Δ) * specific_heat(nls, T, Δ), Δmin, Δmax )[1]
end
specific_heat(T, dve::DonutVolcanoEnsemble, Δmin = 0, Δmax = Inf) = specific_heat(TwoLevelSystem(), T, dve, Δmin, Δmax)

"""
    RandomDonutVolcanoGenerator{T <: Real}

Object to generate a random [`DonutVolcanoEnsemble`](@ref).
"""
struct RandomDonutVolcanoGenerator{T <: Real}
    ndist::DiscreteUniform
    μdist::Uniform
    σdist::Uniform

    function RandomDonutVolcanoGenerator{T}( nmax::Int, μmax, σmax, nmin::Int = 1, μmin = 0., σmin = 1e-2 ) where T
        nmin, nmax = promote(nmin, nmax)
        μmin, μmax = T(μmin), T(μmax)
        σmin, σmax = T(σmin), T(σmax)

        # Check inputs
        @assert nmin ≥ one(nmin) "Minimum number of Donut Volcanos must be greater than 1. Got $nmin"
        @assert nmax > nmin "Maximum number of Donut Volcanos must be greater than the minimum. Got ($nmin, $nmax)."
        valid((μmin, σmin))
        valid((μmax, σmax))

        # Construct
        return new{T}( DiscreteUniform(nmin, nmax), Uniform(μmin, μmax), Uniform(σmin, σmax) )
    end

    RandomDonutVolcanoGenerator(args...) = RandomDonutVolcanoGenerator{Float64}(args...)
end

number_distribution(rdveg::RandomDonutVolcanoGenerator) = rdveg.ndist
center_distribution(rdveg::RandomDonutVolcanoGenerator) = rdveg.μdist
width_distribution(rdveg::RandomDonutVolcanoGenerator) = rdveg.σdist

"""
    rand([rng = GLOBAL_RNG], ::RandomDonutVolcanoGenerator{T})

Generate a random [`DonutVolcanoEnsemble`](@ref) of type `T` based on the 
[`RandomDonutVolcanoGenerator`](@ref) supplied. By default, `T = Float64`.

```jldoctest
julia> using Random 

julia> rng = MersenneTwister(42);  # Choice for longevity. Use Xoshiro in practice or the GLOBAL_RNG.

julia> rdveg = RandomDonutVolcanoGenerator(3, 10, 10);

julia> dve = rand(rng, rdveg);     # Suppress output for floating-point error reasons.

julia> for tup ∈ ensemble(dve)
           @show round.( tup; digits = 6 )
       end
round.(tup; digits = 6) = (6.23099, 2.780566)
round.(tup; digits = 6) = (9.745867, 4.499406)
round.(tup; digits = 6) = (8.427712, 3.660781)
```
"""
function Base.rand(rng::AbstractRNG, rdveg::RandomDonutVolcanoGenerator{T}) where T
    num_donuts = rand(rng, number_distribution(rdveg))
    dve = DonutVolcanoEnsemble(T)
    for idx ∈ UnitRange(1, num_donuts)
        μ = rand(rng, center_distribution(rdveg))
        σ = rand(rng, width_distribution(rdveg))
        push!(dve, (μ, σ))
    end
    return dve
end
Base.rand(rdveg::RandomDonutVolcanoGenerator) = Base.rand(Random.GLOBAL_RNG, rdveg)

"""
    rand([rng = GLOBAL_RNG], ::RandomDonutVolcanoGenerator{T}, num_ensembles::Int)

Return a `Vector` of random [`DonutVolcanoEnsemble`](@ref)s of length `num_ensembles`.
The type of each [`DonutVolcanoEnsemble`](@ref) is `T`. By default, the `GLOBAL_RNG` is
used.
"""
function Base.rand(rng::AbstractRNG, rdveg::RandomDonutVolcanoGenerator{T}, num_ensembles::Int) where T
    output = Vector{DonutVolcanoEnsemble{T}}(undef, num_ensembles)
    for idx ∈ UnitRange(1, num_ensembles)
        output[idx] = rand(rng, rdveg)
    end
    return output
end
Base.rand(rdveg::RandomDonutVolcanoGenerator, num_ensembles::Int) = Base.rand(Random.GLOBAL_RNG, rdveg, num_ensembles)