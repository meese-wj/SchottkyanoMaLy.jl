
using NumericalIntegration

export gausskernel

@doc raw"""
    mass2( xdata, ydata, [method = TrapezoidalFast()] )

Use `NumericalIntegration.jl` to compute the "square mass" of a set of data
`ydata` over the domain `xdata`. 

!!! note
    The default `IntegrationMethod` is set to `TrapezoidalFast` which will lead 
    to errors if `xdata` and `ydata` are of different lengths. For experimental
    data inputs, however, this should be fine because the specific heat is 
    measured for a specific temperature.

The square-mass formula, given as a continuous integral, is 

```math
M^2[y(x)] = \int_a^b \mathrm{d}x\, \left[ y(x) \right]^2.
```
"""
mass2( xdata, ydata, method::IntegrationMethod = TrapezoidalFast() ) = NumericalIntegration.integrate( xdata, ydata .^ 2, method ) 
@doc raw"""
    dist2(xdata, y1data, y2data, [method = TrapezoidalFast()])

Compute the square-distance between two sets of `ydata` on the given `xdata` domain.
This function calls [`mass2`](@ref) to perform the integration. 

The formula for the square distance between functions ``y_1`` and ``y_2`` on the 
continuous interval `(a,b)` is given by 

```math
d^2(y_1, y_2) = M^2[y_1(x) - y_2(x)] = \int_a^b \mathrm{d}x\, \left[ y_1(x) - y_2(x) \right]^2.
```
"""
dist2(xdata, y1, y2, method::IntegrationMethod = TrapezoidalFast()) = mass2(xdata, y1 .- y2, method)
@doc raw"""
    gausskernel(xdata, y1data, y2data, hypσ, [method = TrapezoidalFast()])

Compute the Gaussian kernel function for two sets of `ydata`, given the hyperparameter `hypσ`. 
This kernel function is of the form

```math
K[y_1(x), y_2(x); \sigma] = \exp\left[ - \frac{d^2[y_1(x), y_2(x)]}{2\sigma^2} \right],
```

where ``d^2`` is the [`dist2`](@ref) functional.
"""
gausskernel(xdata, y1, y2, hypσ, method::IntegrationMethod = TrapezoidalFast()) = exp( -dist2(xdata, y1, y2, method) / (2 * hypσ^2) )
