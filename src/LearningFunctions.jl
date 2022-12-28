
using LinearAlgebra

single_component_deviation(prediction, value) = @fastmath @. prediction - value
single_component_loss(prediction, value) = @fastmath @. 0.5 * single_component_deviation(prediction, value)^2

@doc raw"""
    single_component_loss_gradient(prediction, value, νT_Minv, Qmat, Minv, fcomp_vector, ∂σd2_ν_vector)

Compute the loss gradient for a single component (taken analytically of [`single_component_loss`](@ref)). 
    
This is a hefty function, so it is best to pass in as many of its parts as possible to avoid multiple computations of the same quantity.

The mathematics for this gradient, as a function of the hyperparameters ``\sigma`` and ``\lambda`` follows as 

```math
\nabla \ell_m = (f^m - \tilde{f}^m) \begin{pmatrix} \partial_\sigma \tilde{f}^m \\ \partial_\lambda \tilde{f}^m \end{pmatrix}, 
```

with 

```math
\tilde{f}^m = \boldsymbol{\nu}^{\mathrm{T}}(\sigma) \left[ K(\sigma) + \lambda \mathbb{1} ]^{-1} \boldsymbol{f}^m.
```

``\boldsymbol{\nu}`` is the ``N_{\mathrm{interp}} \times 1`` `Vector` comprised of the [`gausskernel`](@ref) calculated between
the input function and the ``N_{\mathrm{interp}}`` functions in the interpolation set. We define the `Matrix` ``M(\sigma, \lambda) \equiv K(\sigma) + \lambda \mathbb{1}``,
such that the predicted component ``\tilde{f}^m`` simplifies to 

```math
\tilde{f}^m = \boldsymbol{\nu}^{\mathrm{T}}(\sigma) M^{-1}(\sigma, \lambda) \boldsymbol{f}^m.
```

The quantity ``\boldsymbol{f}^m`` is the ``N_{\mathrm{interp}} \times 1`` `Vector` of the ``m^{\mathrm{th}}`` Chebyshev component of each ensemble in 
the interpolation set.

Recalling that 

```math
\partial (M^{-1}) = - M^{-1} (\partial M) M^{-1},
```

it follows 

```math
\partial_\lambda \tilde{f}^m = - \boldsymbol{\nu}^{\mathrm{T}} M^{-1} M^{-1} \boldsymbol{f}^m,
```

and 

```math
\partial_\sigma \tilde{f}^m = -\left[ \sigma^{-3}(d^2\boldsymbol{\nu})^{\mathrm{T}} + \boldsymbol{\nu}^{\mathrm{T}}M^{-1} Q(\sigma, \lambda) \right] M^{-1}\boldsymbol{f}^m.
```

The `Vector` ``\sigma^{-3}(d^2 \boldsymbol{\nu})`` is the element-wise derivative of ``\boldsymbol{\nu}`` and the matrix ``Q(\sigma, \lambda)`` is that for ``K(\sigma, \lambda)``.
"""
function single_component_loss_gradient( prediction, value, νT_Minv, Qmat, Minv, fcomp_vector, ∂σd2_ν_vector )
    Minv_f_vector = Minv * fcomp_vector
    ∂σval = ( transpose(∂σd2_ν_vector) + νT_Minv * Qmat ) * Minv_f_vector
    ∂λval = KinT_Minv * Minv_f_vector
    return -single_component_deviation(prediction, value) .* ( ∂σval[1], ∂λval[1] )
end

function total_loss( predictions, values )
    return sum( single_component_loss(predictions, values) )
end
