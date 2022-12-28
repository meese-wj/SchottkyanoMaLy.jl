
using LinearAlgebra
using NumericalIntegration
using FastChebInterp

export compute_cVs!, compute_cVs,
       input_test_msqdiffs, input_test_msqdiffs!,
       populate_msqdiffs!, populate_msqdiffs,
       kernel

"""
    compute_cVs!( cV_arr, ensembles, temps; [kwargs...] )

Compute the [`specific_heat`](@ref) at each temperature in `temps` for 
a given `ensemble` as a single column. By assumption, `cV_arr` is a 
`length(temps) × length(ensembles)` `Matrix`, and it has been initialized
with `zeros`.

# Keyword arguments

* `nls = `[`TwoLevelSystem()`](@ref): The type of [`NLevelSystem`](@ref) used to compute the [`specific_heat`](@ref).
* `Δmin = 0.`: The minimum value of the level-splitting in the integration over each `ensemble`.
* `Δmax = Inf`: The maximum value of the level-splitting in the integration over each `ensemble`.
* `rtol = sqrt(eps())`: The relative tolerance used in `QuadGK.quadgk` to integrate over the `ensembles`.

!!! note 
    Since this populates the specific heat values with `QuadGK.quadgk`, it is best only to compute once
    as it's a bit of an expensive operation.
"""
function compute_cVs!( cV_arr, ensembles, temps; nls = TwoLevelSystem(), Δmin = 0., Δmax = Inf, rtol = sqrt(eps()) )
    @inbounds Threads.@threads for idx ∈ eachindex(ensembles)
        cV_arr[:, idx] .= specific_heat(nls, temps, ensembles[idx], Δmin, Δmax; rtol = rtol)   
    end
    return cV_arr
end
"""
    compute_cVs(ensembles, temps; [kwargs...])

Creates a zero `Matrix` of size `length(temps) × length(ensembles)` and 
passes it to [`compute_cVs!`](@ref).
"""
function compute_cVs(ensembles, temps; kwargs...)
    cV_arr = zeros( length(temps), length(ensembles) )
    return compute_cVs!(cV_arr, ensembles, temps; kwargs...)
end
"""
    input_test_msqdiffs!(output, test_cV, cV_arr, temps, [method = TrapezoidalFast()])

Compute a single column vector of the [`msqdiff`](@ref)s between the `test_cV`
argument and each column of the `cV_arr`, all measured as functions of `temps` 
temperature. Store the result in the `output` vector.
"""
function input_test_msqdiffs!(output, test_cV, cV_arr, temps; method = TrapezoidalFast())
    @assert length(test_cV) == size(cV_arr)[1] == length(temps) "Size mismatch. Each specific heat must have the same number of temperatures ($(length(temps))). Got $(length(test_cV)) and $(size(cV_arr)[1])."
    @assert length(output) == size(cV_arr)[2] "Size mismatch. The length of the output vector must be the number of ensembles. Got $(length(output)) and $(size(cV_arr)[2])."
    @inbounds @simd for idx ∈ eachindex(output)
        output[idx] = msqdiff(temps, view(cV_arr, :, idx), test_cV, method)
    end
    return output
end
"""
    input_test_msqdiffs(test_cV, cV_arr, temps, [method = TrapezoidalFast()])

Allocate an appropriate `output` `Vector` and pass it to [`input_test_msqdiffs!`](@ref).
"""
function input_test_msqdiffs(test_cV, cV_arr, temps; method = TrapezoidalFast())
    output = zeros(eltype(test_cV), size(cV_arr)[2])
    return input_test_msqdiffs!(output, test_cV, cV_arr, temps; method = method)
end
"""
    populate_msqdiffs!( msqdiff_mat, cV_arr, temps; [kwargs...])

Compute the [`msqdiff`](@ref) for each pair of specific heats in `cV_arr` as 
a function of temperature (`temps`) as the `xdata`. This assumes that 

1. `msqdiff_mat` is a `N × N` zero `Matrix`,
2. `N == size(cV_arr)[2]`, where `cV_arr` is a specific heat `Matrix` calculated from [`compute_cVs!`](@ref),
3. the number of temperatures in `temps` equals the number of specific heat rows.

This function uses [`input_test_msqdiffs!`](@ref) to fill the `Matrix`. The `input` is the 
specific heat of a given ensemble, and the `test`s are those from all other ensembles.

!!! note
    `N` is likely to be a large number, so it is best to use this function once.
"""
function populate_msqdiffs!( msqdiff_mat, cV_arr, temps; method = TrapezoidalFast() )
    # Not true anymore...
    # This populates the `msqdiff_mat` by exploiting the symmetry in [`msqdiff`](@ref). So 
    # only `N(N+1)/2` elements are computed and then the matrix is symmetrized. (Hence the need
    # for a zero matrix input.)
    @assert size(cV_arr)[1] == length(temps) "Size mismatch. There must be as many temperatures ($(length(temps))) as specific heats ($(size(cV_arr)[1]))."
    @assert size(msqdiff_mat)[1] == size(msqdiff_mat)[2] == size(cV_arr)[2] "Size mismatch. The first argument must be a Matrix of size N × N, where N is the number of columns in the second argument."
    
    NL = size(msqdiff_mat)[1]
    @inbounds Threads.@threads for col ∈ UnitRange(1, NL)
        ycol = @view cV_arr[:, col]
        input_test_msqdiffs!(view(msqdiff_mat, :, col), ycol, cV_arr, temps; method = method)
    end
    # TODO: Should I keep this method or rely on input_test_msqdiffs?
    #     msqdiff_mat[col, col] = 0.5 * msqdiff( temps, ycol, ycol, method ) # half because it will be added back in the transpose
    #     @simd for row ∈ UnitRange(col + oneunit(col), NL)
    #         yrow = @view cV_arr[:, row]
    #         msqdiff_mat[row, col] = msqdiff(temps, yrow, ycol, method)
    #     end
    # end
    # msqdiff_mat .= msqdiff_mat .+ transpose(msqdiff_mat)
    return msqdiff_mat
end
"""
    populate_msqdiffs(cV_arr, temps; [kwargs...])

Creates a zero `Matrix` of size `length(ensembles) × length(ensembles)` and 
passes it to [`populate_msqdiffs!`](@ref).
"""
function populate_msqdiffs(cV_arr, temps; kwargs...)
    N = size(cV_arr)[2]
    mat = zeros( N, N )
    return populate_msqdiffs!(mat, cV_arr, temps; kwargs...)
end
"""
    kernel(msqdiff_matrix, hypσ)

Compute the element-wise [`gausskernel`](@ref) of the `msqdiff_matrix`
with respect to the hyperparameter `hypσ`. This makes a copy.
"""
kernel(mat, hypσ) = gausskernel.(mat, hypσ)