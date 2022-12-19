
export regularized_inverse, minimizing_component, minimizing_solution

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
minimizing_component( kernel_matrix_prefactor, train_component_vector ) = return (kernel_matrix_prefactor * train_component_vector)[begin]
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