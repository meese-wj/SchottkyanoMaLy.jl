
using LinearAlgebra
using Statistics

export single_component_loss, single_component_loss_gradient, mean_component_loss_gradient, total_loss_gradient, total_loss

single_component_deviation(prediction, value) = @fastmath @. prediction - value
single_component_loss(prediction, value) = @fastmath @. 0.5 * single_component_deviation(prediction, value)^2

@doc raw"""
    single_component_loss_gradient(prediction, value, νvector, Qmat, Minv, fcomp_vector, ∂σd2_ν_vector)

Compute the loss gradient for a single component (taken analytically of [`single_component_loss`](@ref)). 
    
This is a hefty function, so it is best to pass in as many of its parts as possible to avoid multiple computations of the same quantity.

The mathematics for this gradient, as a function of the hyperparameters ``\sigma`` and ``\lambda`` follows as 

```math
\nabla \ell_m = (f^m - \tilde{f}^m) \begin{pmatrix} \partial_\sigma \tilde{f}^m \\ \partial_\lambda \tilde{f}^m \end{pmatrix}, 
```

with 

```math
\tilde{f}^m = \boldsymbol{\nu}^{\mathrm{T}}(\sigma) \left[ K(\sigma) + \lambda \mathbb{1} \right]^{-1} \boldsymbol{f}^m.
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
function single_component_loss_gradient( prediction::AbstractFloat, value::AbstractFloat, νvector, ∂σνvector, Qmat, Minv, fcomp_vector )
    Minv_f_vector = Minv * fcomp_vector # column vector
    νT_Minv = transpose(νvector) * Minv # row vector
    ∂σval  = dot( ∂σνvector, Minv_f_vector) # column ⋅ column
    ∂σval += -dot( νT_Minv, Qmat, Minv_f_vector) # column ⋅ matrix ⋅ column
    ∂λval = -dot( νT_Minv, Minv_f_vector ) # row ⋅ column
    return single_component_deviation(prediction, value) .* ( ∂σval, ∂λval ) # return a Tuple as the gradient ∇ = (∂σ, ∂λ)
end
function single_component_loss_gradient( component_idx::Int, ensemble_idx::Int, updated_trainset, updated_interpset )
    pred = predicted_components(updated_trainset)[component_idx, ensemble_idx]
    val  = cheby_components(updated_trainset)[component_idx, ensemble_idx]
    return single_component_loss_gradient(pred, val, 
                                          gausskernel_TvI(updated_trainset, ensemble_idx), gausskernel_deriv_TvI(updated_trainset, ensemble_idx),
                                          deriv_Gram(updated_interpset), inv_Gram(updated_interpset), cheby_components(updated_interpset, component_idx) )
end

function mean_component_loss_gradient( component_idx, updated_trainset, updated_interpset )
    num_ens::Int = num_ensembles(updated_trainset)
    out_type = Base.eltype(updated_trainset)
    output_σ::out_type = zero(out_type)
    output_λ::out_type = zero(out_type)
    @inbounds for ens_idx ∈ UnitRange(1, num_ens)
        output = single_component_loss_gradient(component_idx, ens_idx, updated_trainset, updated_interpset)
        output_σ += output[1]
        output_λ += output[2]
    end
    return (output_σ, output_λ) ./ num_ens
end

function total_loss_gradient( updated_trainset, updated_interpset, num_comps = num_cheby_components(updated_interpset) )
    @show num_comps
    out_type = Base.eltype(updated_trainset)
    output_σ::out_type = zero(out_type)
    output_λ::out_type = zero(out_type)
    @inbounds for comp_idx ∈ UnitRange(1, num_comps)
        output = mean_component_loss_gradient(comp_idx, updated_trainset, updated_interpset)
        output_σ += output[1]
        output_λ += output[2]
    end
    return (output_σ, output_λ)
end

function total_loss( predictions, values, num_comps::Union{Nothing, Int} = nothing )
    num_comps === nothing ? size(predictions)[1] : num_comps
    preds = @view predictions[1:num_comps, :]
    vals = @view values[1:num_comps, :]
    return sum( tup -> (mean ∘ single_component_loss)(tup...), zip( eachcol.( (preds, vals) )... ) )
end
