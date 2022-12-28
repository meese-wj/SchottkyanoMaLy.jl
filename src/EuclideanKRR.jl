
export regularized_inverse, minimizing_component, minimizing_solution, InterpolationSet, TrainingSet

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

"""
    InterpolationSet{T <: Real}

Container around the relevant arrays used for the interpolations in
Kernel Ridge Regression. 


# Contents

It is assumed that the order of the Chebyshev interpolation is `n` and the 
number of ensembles in the interpolation set is `N`.

* `cheby_components::Matrix{T}`: an `N × (n + 1)` matrix of the Chebyshev coefficients for all ensembles in the `InterpolationSet`.
* `msqdiff_Gram::Matrix{T}`: an `N × N` matrix of the [`msqdiff`](@ref) values between each of the ensembles in the `InterpolationSet`.
* `inv_Gram::Matrix{T}`: an `N × N` matrix used as a temporary placeholder for the intermediary calculations, for example, the inverse of the regularized Gram matrix. 
"""
struct InterpolationSet{T <: Real}
    cheby_components::Matrix{T}
    msqdiff_Gram::Matrix{T}
    inv_Gram::Matrix{T}
end

function InterpolationSet( _cheby_components, _msqdiff_Gram )
    @assert size(_cheby_components)[1] == size(_msqdiff_Gram)[1] == size(_msqdiff_Gram)[2] "Size mismatch. Got $(size(_cheby_components)) and $(size(_msqdiff_Gram))."
    new_type = promote_type( eltype(_cheby_components), eltype(_msqdiff_Gram) )
    return InterpolationSet{new_type}( new_type.(_cheby_components), new_type.(_msqdiff_Gram), zeros(new_type, size(_msqdiff_Gram)...) )
end

cheby_components(intset::InterpolationSet, order) = view( intset.cheby_components, :, order )
msqdiff_Gram(intset::InterpolationSet) = view(intset.msqdiff_Gram, :, :)
inv_Gram(intset::InterpolationSet) = view(intset.inv_Gram, : ,:)

"""
    TrainingSet{T <: Real}

Container around the relevant arrays used for the training of the 
regularized Kernel Ridge Regression model.
"""
struct TrainingSet{T <: Real}
    cheby_components::Matrix{T}
    predicted_components::Matrix{T}
    msqdiff_interpolation::Matrix{T}
end