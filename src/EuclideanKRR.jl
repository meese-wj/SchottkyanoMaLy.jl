
export minimizing_component

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
    return transpose(kernel_column) * inv_kernel_matrix * train_component_vector
end