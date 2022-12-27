
using NumericalIntegration

export msqdiff, gausskernel, ∂σ_gausskernel

@doc raw"""
    mass2( xdata, ydata, [method = TrapezoidalFast()] )

Use `NumericalIntegration.jl` to compute the "mean-square-mass" of a set of data
`ydata` over the domain `xdata`. 

!!! note
    The default `IntegrationMethod` is set to `TrapezoidalFast` which will lead 
    to errors if `xdata` and `ydata` are of different lengths. For experimental
    data inputs, however, this should be fine because the specific heat is 
    measured for a specific temperature.

The mean-square-mass formula, given as a continuous integral, is 

```math
m^2[y(x)] = \frac{1}{b - a}\int_a^b \mathrm{d}x\, \left[ y(x) \right]^2.
```

```jldoctest
julia> xvals = LinRange(1, 10, 10);

julia> yvals = sqrt.(xvals);  # In this case the trapezoidal rule is exact

julia> value = SchottkyAnoMaLy.mass2(xvals, yvals)
5.5

julia> value ≈ 1/(10 - 1) * 1/2 * ( maximum(xvals)^2 - minimum(xvals)^2 )
true
```
"""
mass2( xdata, ydata, method::IntegrationMethod = TrapezoidalFast() ) = 1 / (xdata[end] - xdata[begin]) * NumericalIntegration.integrate( xdata, ydata .^ 2, method ) 
@doc raw"""
    msqdiff(xdata, y1data, y2data, [method = TrapezoidalFast()])

Compute the mean-square-difference between two sets of `ydata` on the given `xdata` domain.
This function calls [`mass2`](@ref) to perform the integration. 

The formula for the mean-square-difference between functions ``y_1`` and ``y_2`` on the 
continuous interval `(a,b)` is given by 

```math
d^2(y_1, y_2) = m^2[y_1(x) - y_2(x)] = \frac{1}{b - a}\int_a^b \mathrm{d}x\, \left[ y_1(x) - y_2(x) \right]^2.
```
"""
msqdiff(xdata, y1, y2, method::IntegrationMethod = TrapezoidalFast()) = mass2(xdata, y1 .- y2, method)
@doc raw"""
    gausskernel(d2, hypσ, [method = TrapezoidalFast()])
    gausskernel(xdata, y1data, y2data, hypσ, [method = TrapezoidalFast()])

Compute the Gaussian kernel function for two sets of `ydata`, given the hyperparameter `hypσ`. 
This kernel function is of the form

```math
K[y_1(x), y_2(x); \sigma] = \exp\left[ - \frac{d^2[y_1(x), y_2(x)]}{2\sigma^2} \right],
```

where ``d^2`` is the [`msqdiff`](@ref) functional.
"""
gausskernel(d2, hypσ) = @fastmath exp( -d2 / (2 * hypσ^2) )
gausskernel(xdata, y1, y2, hypσ, method::IntegrationMethod = TrapezoidalFast()) = gausskernel( msqdiff(xdata, y1, y2, method), hypσ )

@doc raw"""
    ∂σ_gausskernel(d2, hypσ)  # \partial<TAB>\sigma<TAB>

Calculate the gradient of [`gausskernel`](@ref) with respect to the hyperparameter `hypσ`.
The explicit formula is given by 

```math
\partial_\sigma K(d^2; \sigma) = \sigma^{-3} d^2 K(d^2; \sigma).
```
"""
∂σ_gausskernel(d2, hypσ) = @fastmath d2 / (hypσ^3) * gausskernel(d2, hypσ)