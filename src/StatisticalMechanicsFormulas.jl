
export TwoLevelSystem, specific_heat

"""
    abstract type NLevelSystem  end

Create a supertype for all non-interacting systems with discrete 
energy levels. 

Each new concrete subtype *must* define a [`specific_heat`](@ref) function.
"""
abstract type NLevelSystem end
"""
    struct TwoLevelSystem end

Empty `struct` for the case of the two-level system. Subtype of [`NLevelSystem`](@ref).
Used to define traits for this specific type of system.
"""
struct TwoLevelSystem <: NLevelSystem end

# Throw a MethodError unless the <: NLevelSystem has an implemented method.
specific_heat(t::Type{<: NLevelSystem}, args...) = throw(MethodError(specific_heat, (t, args...)))

@doc raw"""
    specific_heat(::Type{TwoLevelSystem}, T, Δ)

Define the specific heat function for a two-level system at temperature `T`
and level-splitting `Δ` (measured in units of temperature). The formula is given by 

```math
c_V(T;\, \Delta) = \left( \frac{\Delta}{T} \right)^2 \mathrm{sech}^2 \left( \frac{\Delta}{T} \right)^2.
```

```jldoctest
julia> specific_heat(TwoLevelSystem, 1, 1)
0.41997434161402614
```
"""
specific_heat(::Type{TwoLevelSystem}, T, Δ) = @. ( Δ / T * sech( Δ / T ) )^2
"""
    specific_heat(T, Δ) = specific_heat(TwoLevelSystem, T, Δ)

The default specific heat implementation chooses a [`TwoLevelSystem`](@ref).

```jldoctest
julia> specific_heat(1, 1)
0.41997434161402614

julia> specific_heat(TwoLevelSystem, 1, 1)
0.41997434161402614
```
"""
specific_heat(T, Δ) = specific_heat(TwoLevelSystem, T, Δ)
