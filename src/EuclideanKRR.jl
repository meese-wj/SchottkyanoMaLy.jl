import Base: eltype

export regularized_inverse, minimizing_component, minimizing_solution, 
       eltype, specific_heats, num_ensembles, num_temperatures,
       InterpolationSet, cheby_components, msqdiff_Gram, deriv_Gram, inv_Gram,
       TrainingSet

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

"""
    InterpolationSet{T <: Real} <: MLMatrixContainer{T}

Container around the relevant arrays used for the interpolations in
Kernel Ridge Regression. 


# Contents

It is assumed that the order of the Chebyshev interpolation is `n`, the 
number of ensembles in the interpolation set is `N`, and the number of temperatures
for the specific heat is `L`.

* `specific_heats::Matrix{T}`: An `L × N` matrix containing all the specific heat values for the ensembles in the `InterpolationSet`.
* `cheby_components::Matrix{T}`: an `N × (n + 1)` matrix of the Chebyshev coefficients for all ensembles in the `InterpolationSet`.
* `msqdiff_Gram::Matrix{T}`: an `N × N` matrix of the [`msqdiff`](@ref) values between each of the ensembles in the `InterpolationSet`.
* `deriv_Gram::Matrix{T}`: an `N × N` matrix of the [`msqdiff`](@ref) values between each of the ensembles in the `InterpolationSet` element-wise multiplied by those from the full kernel Gram matrix.
* `inv_Gram::Matrix{T}`: an `N × N` matrix used as a temporary placeholder for the intermediary calculations, for example, the inverse of the regularized Gram matrix. 
"""
struct InterpolationSet{T <: Real} <: MLMatrixContainer
    specific_heats::Matrix{T}
    cheby_components::Matrix{T}
    msqdiff_Gram::Matrix{T}
    deriv_Gram::Matrix{T}
    inv_Gram::Matrix{T}
end

function InterpolationSet( _all_cVs, _cheby_components, _msqdiff_Gram )
    @assert size(_all_cVs)[2] == size(_cheby_components)[1] "Size mismatch. Got $( size(_all_cVs) ) and $( size(_cheby_coefficients) )"
    @assert size(_cheby_components)[1] == size(_msqdiff_Gram)[1] == size(_msqdiff_Gram)[2] "Size mismatch. Got $(size(_cheby_components)) and $(size(_msqdiff_Gram))."
    new_type = promote_type( eltype(_all_cVs), eltype(_cheby_components), eltype(_msqdiff_Gram) )
    return InterpolationSet{new_type}( new_type.(_all_cVs), 
                                       new_type.(_cheby_components), 
                                       new_type.(_msqdiff_Gram), 
                                       zeros(new_type, size(_msqdiff_Gram)...),
                                       zeros(new_type, size(_msqdiff_Gram)...)
                                    )
end

msqdiff_Gram(intset::InterpolationSet) = intset.msqdiff_Gram
deriv_Gram(intset::InterpolationSet) = intset.deriv_Gram
inv_Gram(intset::InterpolationSet) = intset.inv_Gram

"""
    TrainingSet{T <: Real} <: MLMatrixContainer

Container around the relevant arrays used for the training of the 
regularized Kernel Ridge Regression model. 

# Contents

This structure assumes that there are `M` ensembles in the `TrainingSet` and 
`N` ensembles in the [`InterpolationSet`](@ref). Additionally, it assumes 
the order of the Chebyshev interpolation is `n` while the number of temperatures 
for the training specific heats is `L`.

* `specific_heats::Matrix{T}`: An `L × M` matrix of the specific heat values for each ensemble in the `TrainingSet`.
* `cheby_components::Matrix{T}`: An `(n + 1) × M` `Matrix` of true values of the Chebyshev coefficients for all ensembles in the `TrainingSet`.
* `predicted_components::Matrix{T}`: A container for the `(n + 1) × M` predicted components from KRR.
* `msqdiff_TvI::Matrix{T}`: An `N × M` matrix containing the [`msqdiff`](@ref) between each element in the `TrainingSet` (`T`) and the [`InterpolationSet`](@ref) (`I`).
"""
struct TrainingSet{T <: Real} <: MLMatrixContainer
    specific_heats::Matrix{T}
    cheby_components::Matrix{T}
    predicted_components::Matrix{T}
    msqdiff_TvI::Matrix{T}
end

function TrainingSet( _train_cVs, _cheby_coefficients, _interpset )
    @assert size(_train_cVs)[1] == num_temperatures(_interpset) "Size mismatch. Got $( size(_train_cVs) ) and $( size(specific_heats(_interpset)) )."
    @assert size(_train_cVs)[2] == size(_cheby_coefficients)[2] "Size mismatch. Got $( size(_train_cVs) ) and $( size(_cheby_coefficients) )."
    @assert size(_cheby_coefficients)[1] == size(cheby_components(_interpset))[2] "Size mismatch. Got $( size(_cheby_coefficients) ) and $( size(cheby_components(_interpset)) )."
    new_type = promote_type( eltype.( _train_cVs, _cheby_coefficients, _interpset )... )
    return TrainingSet{new_type}( new_type.(_all_cVs),
                                  new_type.(_cheby_coefficients),
                                  zeros(new_type, size(_cheby_coefficients)...),
                                  zeros(new_type, num_ensembles(_interpset), size(_all_cVs)[2])
    )
end

function TrainingSet(_train_cVs, _temperatures, _cheby_coefficients, _interpset; kwargs...)
    trainset = TrainingSet(_train_cVs, _cheby_coefficients, _interpset)
    @assert length(_temperatures) == num_temperatures(trainset) == num_temperatures(_interpset) "Size mismatch. Got $( length(_temperatures) ), $(num_temperatures(trainset)), and $(num_temperatures(_interpset))."
    # Now populate the msqdiff_TvI matrix
    for train_cidx ∈ eachindex(eachcol(msqdiff_TvI(trainset)))
        train_column = view( msqdiff_TvI(trainset), :, train_cidx )
        input_reference_msqdiffs!( train_column, specific_heats(trainset, train_cidx), specific_heats(_interpset), _temperatures; kwargs... )
    end
    return trainset
end

predicted_components(trainset::TrainingSet) = trainset.predicted_components
predicted_components(trainset::TrainingSet, order) = view( predicted_components(trainset), :, order )
msqdiff_TvI(trainset::TrainingSet) = trainset.msqdiff_TvI