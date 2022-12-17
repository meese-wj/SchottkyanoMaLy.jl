
using StaticArrays
using FastChebInterp

export chebycoefficients!, chebycoefficients, create_chebinterp

@doc raw"""
    chebydecomposition(ensemble, order, Δmin, Δmax, [ coefftol = zero(Float64) ])

Use `FastChebInterp.jl` to compute the Chebyshev interpolation of a given `ensemble`
to a polynomial `order` on the `[Δmin, Δmax]` interval. This function returns the 
coefficients ``a_k`` computed by `FastChebInterp.chebinterp` and defined by 

```math
\mathrm{p}(\Delta) \approx \sum_{k=0}^M a_k T_k(\Delta),
```

where ``\mathrm{p}(\Delta)`` is the `ensemble`, ``M`` is the `order`, and ``T_k(\Delta)``
is the ``k^{\mathrm{th}}`` [Chebyshev polynomial of the first kind](https://en.wikipedia.org/wiki/Chebyshev_polynomials?oldformat=true).

!!! note 
    Changing the optional `coefftol` from the default value of `0.0` may lead to issues
    downstream. Essentially, `FastChebInterp.jl` uses this (I think) for precision purposes,
    but if it is nonzero then the output `order` is not guaranteed to be what is provided as
    and argument.
"""
function chebydecomposition(ensemble, order, Δmin, Δmax, coefftol = zero(Float64))
    xvals = chebpoints(order, Δmin, Δmax)
    cheby = chebinterp(ensemble(xvals), Δmin, Δmax; tol = coefftol)
    # TODO: Allow coefftol != 0.0 and still return an iterable of size order + 1.
    #       Probably just need to set all of the higher orders to zero.
    return cheby.coefs
end
"""
    chebycoefficients!(coefficients, ensembles, order, Δmin, Δmax, [coefftol = zero(Float64)])

Compute all of the Chebyshev `coefficients` for each of the `ensembles` to a given `order` on the 
interval `[Δmin, Δmax]`. This function uses [`chebydecomposition`](@ref) to do so, and assumes the 
`coefficients` argument is a `length(ensembles) × (order + 1)` `Matrix`.

Because this is a somewhat complex calculation, it should probably be calculated once.
"""
function chebycoefficients!(coefficients, ensembles, order, Δmin, Δmax, coefftol = zero(Float64))
    @assert size(coefficients)[1] == length(ensembles) "Size mismatch. There must be as many rows ($(size(coefficients)[1])) in the first argument as ensembles ($(length(ensembles)))."
    @assert size(coefficients)[2] == order + oneunit(order) "Size mismatch. The number of columns ($(size(coefficients)[1])) in the first argument must be equal to the order ($order) of the polynomial plus one."

    @inbounds Threads.@threads for idx ∈ eachindex(ensembles)
        coefs = chebydecomposition( ensembles[idx], order, Δmin, Δmax, coefftol )
        coefficients[idx, :] .= coefs[:]
    end
    return coefficients
end
"""
    chebycoefficients(ensembles, order, Δmin, Δmax, [coefftol = zero(Float64)])

Allocate the appropriate `coefficients` `Matrix` and pass it to [`chebycoefficients!`](@ref).
"""
function chebycoefficients(ensembles, order, Δmin, Δmax, coefftol = zero(Float64))
    coeffs = zeros(eltype(ensembles[begin]), length(ensembles), order + oneunit(order))
    return chebycoefficients!(coeffs, ensembles, order, Δmin, Δmax, coefftol)
end
"""
    create_chebinterp(coefficients, lowerbound, upperbound, [N = 1])

Create a new `FastChebInterp.ChebPoly` from a set of Chebyshev `coefficients`. By default (and
until I can think of a reason to generalize), this will create a 1D `ChebPoly`. By assumption,
the `lowerbound` and `upperbound` are numbers.
"""
function create_chebinterp( coefficients, lowerbound, upperbound, N = 1 )
    lb, ub = promote(lowerbound, upperbound)
    return FastChebInterp.ChebPoly{N, eltype(coefficients), typeof(lb)}( copy(coefficients), SVector{N}(lb), SVector{N}(ub) )
end