
using SpecialFunctions

@doc raw"""
    Theta(x)

Package-specific implementation of the Heaviside Theta function. Our definition is

```math
\Theta(x) = \begin{cases}
1, & x \geq 0
\\
0, & \mathrm{otherwise}
\end{cases}.
```
"""
Theta(x) = map( val -> convert(eltype(val), val ≥ zero(val)), x )

@doc raw"""
    donutvolcano(x, μ, σ)

The `donutvolcano` distribution calculated for an input `x` with center `μ` and width `σ`.
The argument `x` can be a scalar or broadcastable.

The distribution itself is given by 

```math
{\rm donutvolcano}(x; μ, σ) = \frac{x \Theta(x)}{\mathcal{N}(\mu,\sigma)} \exp\left[ -\frac{\left(x - \mu\right)^2}{2\sigma^2} \right],
```

where we use our own Heaviside [`Theta`](@ref) function above. The normalization constant ``\mathcal{N}(\mu,\sigma)`` given by [`norm_donutvolcano`](@ref).
"""
function donutvolcano(x, μ, σ)
	normalizer = norm_donutvolcano(μ, σ)
	return @. x * Theta(x) * exp( -0.5 * ( (x - μ) / σ )^2 ) / normalizer
end

@doc raw"""
    norm_donutvolcano(μ, σ)

The normalization for a [`donutvolcano`](@ref) distribution with center `μ` and width `σ`.
The explicit form is given by

```math
\mathcal{N}(\mu, \sigma) = \sqrt{\frac{\pi }{2}} \mu  \sigma 
   \left[\text{erf}\left(\frac{\mu }{\sqrt{2} \sigma
}\right)+1\right]+\sigma ^2 {\rm e}^{-\frac{\mu ^2}{2 \sigma ^2}}.
```
"""
function norm_donutvolcano(μ, σ)
	arg = μ / (σ * sqrt(2))
	return σ^2 * exp( -arg^2 ) + sqrt(π/2) * μ * σ * (oneunit(μ) + erf(arg))
end