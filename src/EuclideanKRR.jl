import Base: eltype
using LinearAlgebra

export regularized_inverse, minimizing_component, minimizing_solution, 
       eltype, specific_heats, num_ensembles, num_temperatures, update!,
       InterpolationSet, cheby_components, msqdiff_Gram, gausskernel_Gram, deriv_Gram, inv_Gram,
       TrainingSet, predicted_components, msqdiff_TvI, gausskernel_TvI, gausskernel_deriv_TvI

regularized_inverse(kernel_matrix, hypλ) = inv(kernel_matrix + hypλ * one(kernel_matrix))

kernel_matrix_transformation(kernel_vector, inv_kernel_matrix) = transpose(kernel_vector) * inv_kernel_matrix

@doc raw"""
    minimizing_component(kernel_column, inv_kernel_matrix, train_component_vector)

Compute the solution to the kernel ridge regression problem (KRR). This solution 
minimizes the regularized square-error

```math
\mathcal{E}_m = \sum_i \left( f^m_i  - \sum_j K(i, j; \sigma) \alpha^m_j \right)^2 + \lambda \sum_{ij} \alpha_i K(i, J) \alpha_j,
```

where ``K(i, j)`` is the kernel function measured between functions ``i`` and ``j``, ``f^m_i`` is the ``m^{\mathrm{th}}``
component of ``i^{\mathrm{th}}`` training function, ``\sigma`` and ``\lambda`` are the hyperparameters,
and ``\boldsymbol{\alpha}^m`` is related to the ``\boldsymbol{f}^m`` training vector as 

```math
\boldsymbol{\alpha}^m = \left( K(i,j;\sigma) + \lambda \mathbb{1} \right)^{-1} \boldsymbol{f}^m.
```

The prediction that minimizes ``\mathcal{E}_m`` is given by 

```math
f^m_{\mathrm{out}} = \boldsymbol{K}^T_{in} \boldsymbol{\alpha}^m.
```

By direct substitution, this function then computes

```math
f^m_{\mathrm{out}} = \boldsymbol{K}^T_{in} \left( K(i,j;\sigma) + \lambda \mathbb{1} \right)^{-1} \boldsymbol{f}^m,
```

where ``\boldsymbol{K}^T_{in}`` `= kernel_column`, the whole inverse is `inv_kernel_matrix`, and ``\boldsymbol{f}^m`` `= train_component_vector`.
"""
function minimizing_component( kernel_column, inv_kernel_matrix, train_component_vector )
    return minimizing_component( view(kernel_matrix_transformation(kernel_column, inv_kernel_matrix), :, :), train_component_vector )
end
minimizing_component( kernel_matrix_prefactor, train_component_vector ) = (kernel_matrix_prefactor * train_component_vector)[begin]
"""
    minimizing_solution(kernel_column, inv_kernel_matrix, train_component_matrix)

Use [`minimizing_component`](@ref) to compute the full solution to the Kernel Ridge Regression problem.
This assumes that `train_component_matrix` is of size `N × M`, where `N` is the number of elements in 
the training set and `M` is the number of Chebyshev coefficients (`order + 1`).
"""
function minimizing_solution( kernel_column, inv_kernel_matrix, train_component_matrix )
    output_components = zeros( eltype(train_component_matrix), size(train_component_matrix)[2] )
    kernel_prefactor = kernel_matrix_transformation(kernel_column, inv_kernel_matrix)
    for idx ∈ eachindex(output_components)
        output_components[idx] = minimizing_component( view(kernel_prefactor, :, :), view(train_component_matrix, :, idx) )
    end
    return output_components
end

abstract type MLMatrixContainer end
specific_heats(_set::MLMatrixContainer) = _set.specific_heats
specific_heats(_set::MLMatrixContainer, ensemble_idx) = view( specific_heats(_set), :, ensemble_idx )
cheby_components(_set::MLMatrixContainer) = _set.cheby_components
cheby_components(_set::MLMatrixContainer, order) = view( cheby_components(_set), :, order )
Base.eltype(_set::MLMatrixContainer) = Base.eltype(specific_heats(_set))
num_ensembles(_set::MLMatrixContainer) = size(specific_heats(_set))[2]
num_temperatures(_set::MLMatrixContainer) = size(specific_heats(_set))[1]

update!(_set::MLMatrixContainer, args...) = throw(MethodError(update!, (_set, args...))) 

"""
    InterpolationSet{T <: Real} <: MLMatrixContainer{T}

Container around the relevant arrays used for the interpolations in
Kernel Ridge Regression. 


# Contents

It is assumed that the order of the Chebyshev interpolation is `n`, the 
number of ensembles in the interpolation set is `N`, and the number of temperatures
for the specific heat is `L`.

## Runtime constants

These matrices are constants throughout the learning process.

* `specific_heats::Matrix{T}`: An `L × N` matrix containing all the specific heat values for the ensembles in the `InterpolationSet`.
* `cheby_components::Matrix{T}`: an `N × (n + 1)` matrix of the Chebyshev coefficients for all ensembles in the `InterpolationSet`.
* `msqdiff_Gram::Matrix{T}`: an `N × N` matrix of the [`msqdiff`](@ref) values between each of the ensembles in the `InterpolationSet`.

## Runtime containers

These matrices are [`update!`](@ref)-ed throughout the learning process.

* `gausskernel_Gram::Matrix{T}`: an `N × N` matrix of the [`gausskernel`](@ref) values between each of the ensembles in the `InterpolationSet`.
* `deriv_Gram::Matrix{T}`: an `N × N` matrix of the [`msqdiff`](@ref) values between each of the ensembles in the `InterpolationSet` element-wise multiplied by those from the full kernel Gram matrix.
* `inv_Gram::Matrix{T}`: an `N × N` matrix used as a temporary placeholder for the intermediary calculations, for example, the inverse of the regularized Gram matrix. 
"""
struct InterpolationSet{T <: Real} <: MLMatrixContainer
    # Runtime constants
    specific_heats::Matrix{T}
    cheby_components::Matrix{T}
    msqdiff_Gram::Matrix{T}
    # Runtime update!-ed
    gausskernel_Gram::Matrix{T}
    deriv_Gram::Matrix{T}
    inv_Gram::Matrix{T}
end

function InterpolationSet( _all_cVs, _cheby_components, _msqdiff_Gram )
    @assert size(_all_cVs)[2] == size(_cheby_components)[1] "Size mismatch. Got $( size(_all_cVs) ) and $( size(_cheby_coefficients) )"
    @assert size(_cheby_components)[1] == size(_msqdiff_Gram)[1] == size(_msqdiff_Gram)[2] "Size mismatch. Got $(size(_cheby_components)) and $(size(_msqdiff_Gram))."
    new_type = promote_type( map(x -> eltype(x), (_all_cVs, _cheby_components, _msqdiff_Gram) )... )
    return InterpolationSet{new_type}( new_type.(_all_cVs), 
                                       new_type.(_cheby_components), 
                                       new_type.(_msqdiff_Gram), 
                                       zeros(new_type, size(_msqdiff_Gram)...),
                                       zeros(new_type, size(_msqdiff_Gram)...),
                                       zeros(new_type, size(_msqdiff_Gram)...)
                                    )
end

msqdiff_Gram(intset::InterpolationSet) = intset.msqdiff_Gram
gausskernel_Gram(intset::InterpolationSet) = intset.gausskernel_Gram
deriv_Gram(intset::InterpolationSet) = intset.deriv_Gram
inv_Gram(intset::InterpolationSet) = intset.inv_Gram

function _compute_gausskernel_Gram!(intset, hypσ)
    @inbounds @simd for idx ∈ eachindex(msqdiff_Gram(intset))
        gausskernel_Gram(intset)[idx] = gausskernel(msqdiff_Gram(intset)[idx], hypσ)
    end
    return gausskernel_Gram(intset)
end
function _compute_deriv_Gram!(intset, hypσ)
    @inbounds @simd for idx ∈ eachindex(msqdiff_Gram(intset))
        deriv_Gram(intset)[idx] = ∂σ_gausskernel(msqdiff_Gram(intset)[idx], gausskernel_Gram(intset)[idx], hypσ)
    end
    return deriv_Gram(intset)
end
function _compute_inv_Gram!(intset, hypλ) 
    Mmat = gausskernel_Gram(intset) + ( hypλ * LinearAlgebra.I )( num_ensembles(intset) )
    inv_Gram(intset) .= inv(Mmat)
end

function update!(intset::InterpolationSet, hypσ, hypλ)
    _compute_gausskernel_Gram!(intset, hypσ)
    _compute_deriv_Gram!(intset, hypσ)
    _compute_inv_Gram!(intset, hypλ)
    return intset
end

"""
    TrainingSet{T <: Real} <: MLMatrixContainer

Container around the relevant arrays used for the training of the 
regularized Kernel Ridge Regression model. 

# Contents

This structure assumes that there are `M` ensembles in the `TrainingSet` and 
`N` ensembles in the [`InterpolationSet`](@ref). Additionally, it assumes 
the order of the Chebyshev interpolation is `n` while the number of temperatures 
for the training specific heats is `L`.

## Runtime constants

These matrices are constants throughout the learning process.

* `specific_heats::Matrix{T}`: An `L × M` matrix of the specific heat values for each ensemble in the `TrainingSet`.
* `cheby_components::Matrix{T}`: An `(n + 1) × M` `Matrix` of true values of the Chebyshev coefficients for all ensembles in the `TrainingSet`.
* `msqdiff_TvI::Matrix{T}`: An `N × M` matrix containing the [`msqdiff`](@ref) between each element in the `TrainingSet` (`T`) and the [`InterpolationSet`](@ref) (`I`).

## Runtime containers

These matrices are [`update!`](@ref)-ed throughout the learning process.

* `gausskernel_TvI::Matrix{T}`: An `N × M` matrix containing the [`gausskernel`](@ref) between each element in the `TrainingSet` (`T`) and the [`InterpolationSet`](@ref) (`I`).
* `gausskernel_deriv_TvI::Matrix{T}`: An `N × M` matrix containing the element-wise derivatives of the [`gausskernel`](@ref) between each element in the `TrainingSet` (`T`) and the [`InterpolationSet`](@ref) (`I`).
* `predicted_components::Matrix{T}`: A container for the `(n + 1) × M` predicted components from KRR.
"""
struct TrainingSet{T <: Real} <: MLMatrixContainer
    # Runtime constants
    specific_heats::Matrix{T}
    cheby_components::Matrix{T}
    msqdiff_TvI::Matrix{T}
    # Runtime update!-ed
    gausskernel_TvI::Matrix{T}
    gausskernel_deriv_TvI::Matrix{T}
    predicted_components::Matrix{T}
end

function TrainingSet( _train_cVs, _cheby_coefficients, _interpset )
    @assert size(_train_cVs)[1] == num_temperatures(_interpset) "Size mismatch. The number of rows of the specific heats should match those in the InterpolationSet. Got $( size(_train_cVs)[1] ) and $( size(specific_heats(_interpset))[1] )."
    @assert size(_train_cVs)[2] == size(_cheby_coefficients)[2] "Size mismatch. The number of columns of the specific heats should match those of the Chebyshev components. Got $( size(_train_cVs)[2] ) and $( size(_cheby_coefficients)[2] )."
    @assert size(_cheby_coefficients)[1] == size(cheby_components(_interpset))[2] "Size mismatch. The number of rows of the Chebyshev components should match the *columns*  of those in the InterpolationSet. Got $( size(_cheby_coefficients)[1] ) and $( size(cheby_components(_interpset))[2] )."
    new_type = promote_type( map( x -> eltype(x), (_train_cVs, _cheby_coefficients, _interpset) )... )
    return TrainingSet{new_type}( new_type.(_train_cVs),
                                  new_type.(_cheby_coefficients),
                                  zeros(new_type, num_ensembles(_interpset), size(_train_cVs)[2]),
                                  zeros(new_type, num_ensembles(_interpset), size(_train_cVs)[2]),
                                  zeros(new_type, num_ensembles(_interpset), size(_train_cVs)[2]),
                                  zeros(new_type, size(_cheby_coefficients)...)
    )
end

function TrainingSet(_temperatures, _train_cVs, _cheby_coefficients, _interpset; kwargs...)
    trainset = TrainingSet(_train_cVs, _cheby_coefficients, _interpset)
    @assert length(_temperatures) == num_temperatures(trainset) == num_temperatures(_interpset) "Size mismatch. The number of temperatures should match the rows of the specific heats (both the training and InterpolationSet). Got $( length(_temperatures) ), $(num_temperatures(trainset)), and $(num_temperatures(_interpset))."
    # Now populate the msqdiff_TvI matrix
    @inbounds Threads.@threads for train_cidx ∈ eachindex(eachcol(msqdiff_TvI(trainset)))
        train_column = view( msqdiff_TvI(trainset), :, train_cidx )
        input_reference_msqdiffs!( train_column, specific_heats(trainset, train_cidx), specific_heats(_interpset), _temperatures; kwargs... )
    end
    return trainset
end

predicted_components(trainset::TrainingSet) = trainset.predicted_components
predicted_components(trainset::TrainingSet, ensemble_idx) = view( predicted_components(trainset), :, ensemble_idx )
msqdiff_TvI(trainset::TrainingSet) = trainset.msqdiff_TvI
gausskernel_TvI(trainset::TrainingSet) = trainset.gausskernel_TvI
gausskernel_TvI(trainset::TrainingSet, ensemble_idx) = view( gausskernel_TvI(trainset), :, ensemble_idx )
gausskernel_deriv_TvI(trainset::TrainingSet) = trainset.gausskernel_deriv_TvI
gausskernel_deriv_TvI(trainset::TrainingSet, ensemble_idx) = view( gausskernel_deriv_TvI(trainset), :, ensemble_idx )

function _compute_gausskernel_TvI!(trainset::TrainingSet, hypσ)
    @inbounds @simd for idx ∈ eachindex(gausskernel_TvI(trainset))
        gausskernel_TvI(trainset)[idx] = gausskernel(msqdiff_TvI(trainset)[idx], hypσ)
    end
    return gausskernel_TvI(trainset)
end
function _compute_deriv_TvI!(trainset::TrainingSet, hypσ)
    @inbounds @simd for idx ∈ eachindex(gausskernel_deriv_TvI(trainset))
        gausskernel_deriv_TvI(trainset)[idx] = ∂σ_gausskernel(msqdiff_TvI(trainset)[idx], gausskernel_TvI(trainset)[idx], hypσ) 
    end
end
function _compute_predicted_components!(trainset::TrainingSet, ensemble_idx, updated_interpset)
    Minv = inv_Gram(updated_interpset)
    interp_components = cheby_components(updated_interpset)
    train_interp_column = view( gausskernel_TvI(trainset), :, ensemble_idx )
    for comp_idx ∈ eachindex( eachcol(interp_components) )
        interp_comp_col = @view interp_components[:, comp_idx]
        predicted_components(trainset, ensemble_idx)[comp_idx] = dot( train_interp_column, Minv, interp_comp_col )
    end
    return predicted_components(trainset, ensemble_idx)
end
function _compute_predicted_components!(trainset::TrainingSet, updated_interpset)
    for train_ens_idx ∈ eachindex(eachcol( predicted_components(trainset) ))
        _compute_predicted_components!( trainset, train_ens_idx, updated_interpset )
    end
    return predicted_components(trainset)
end

function update!(trainset::TrainingSet, updated_intset::InterpolationSet, hypσ)
    _compute_gausskernel_TvI!(trainset, hypσ)
    _compute_deriv_TvI!(trainset, hypσ)
    _compute_predicted_components!(trainset, updated_intset) 
    return trainset
end