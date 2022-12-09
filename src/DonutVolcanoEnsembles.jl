
using SpecialFunctions
import Base: eltype, push!, append!

# Base overloads
export eltype, push!, append!
# DonutVolcanoEnsemble stuff
export donutvolcano, DonutVolcanoEnsemble, ensemble, μvalue, σvalue

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
