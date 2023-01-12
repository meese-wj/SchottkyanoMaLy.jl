
using Random
using Optim
using JLD2
import Base: eltype

export SchottkyOptions, eltype, SchottkyAnalysis, train!, predict!, create_chebinterp, save_analysis, load_analysis

const _SA_input_pair = Tuple{T, T} where T # This is purely in anticipation of maybe adding Vector pair too for convenience

"""
    SchottkyOptions{T <: AbstractFloat}

Keyword-constructed `struct` that handles all of the default 
configurable options for the SchottkyAnalysis. 

# Usage

The simplest usage of `SchottkyOptions` is just to use the defaults with `T = Float64`. Note that the `distribution_domain`
always must be supplied since it is set by the temperature scale in the problem.

```jldoctest
julia> opts = SchottkyOptions(distribution_domain = (0., 20.)); # Use the default values and type Float64

julia> eltype(opts)
Float64
```

However, if a different float is required, say `Float32`, just pass the desired type into the constructor as:

```jldoctest
julia> opts32 = SchottkyOptions(distribution_domain = (0., 20.), analysis_type = Float32); # Use the default values but set the analysis_type to Float32

julia> eltype(opts32)
Float32
```

Finally, if one wants to adjust any of the options to a different value, use the keyword in the 
constructor.

```jldoctest
julia> opts = SchottkyOptions(distribution_domain = (0., 20.), analysis_seed = 25); # Keep all default values (T = Float64) except the analysis_seed which we set to 25

julia> opts.analysis_seed
25

julia> opts32 = SchottkyOptions(distribution_domain = (0., 20.), analysis_type = Float32, analysis_seed = 25); # Do the same but for T = Float32

julia> opts32.analysis_seed
25
```

# Options

These options are organized by purpose.

## Analysis-wide Options

These parameters are for configuring the analysis globally.

* `analysis_seed = 42`: global random seed to set for reproducibility
* `analysis_rng = Xoshiro(analysis_seed)`: global random number generator for reproducibility
* `analysis_nls = TwoLevelSystem()`: the type of statistical mechanics model used to form the specific heats from a barrier distribution
* `analysis_iterations = 1`: the number of times one wants to perform the analysis for different [`InterpolationSet`](@ref) and [`TrainingSet`](@ref) and average the predictions

!!! note 
    If, for some reason, `analysis_rng` does not match `analysis_seed`, don't fret. I'll overwrite 
    whatever the seed was for the passed `analysis_rng` upon construction.


## Machine-Learning Algorithm Options

These are for the specific machine-learning algorithm that [`SchottkyAnoMaLy`](@ref)
implements.

* `num_learn = 128`: the number of ensembles in the [`InterpolationSet`](@ref)
* `num_train = num_interp`: the number of ensembles in the [`TrainingSet`](@ref)
* `max_cheby_component = 10`: the maximum component to consider in the [`total_loss`](@ref) when training the hyperparameters
* `initial_hyperparameters = (0.5, 0.5)`: the kickstart value for the `(σ, λ)` hyperparameters in [`GaussianKRRML`](@ref) model

### Optimization Options

These options are used by `Optim.jl` to tune the hyperparameters on the [`TrainingSet`](@ref).
Currently, we use a multivariate constrained local optimization method since 

1. only the magnitude of the interpolation bandwidth `σ` matters,
2. the stiffness `λ > 0` to ensure boundedness of the solution and numerical stability.

Local optimization is preferred over global optimization in this case, as we just need a "good enough" prediction, as I recommend
repeating this process for a few different [`InterpolationSet`](@ref)s and [`TrainingSet`](@ref)s to build confidence in the results.

* `optim_lb = (1e-12, 1e-8)`: the lower `(σ, λ)` bounds for the constrained optimization
* `optim_ub = (Inf, Inf):` the upper `(σ, λ)` bounds for the constrained optimization
* `optim_method = Optim.LBFGS()`: the multivariate *unconstrained* optimization method
* `optim_algorithm = Optim.Fminbox(optim_method)`: the multivariate *constrained* optimization algorithm used on top of `optim_method`
* `optim_options = Optim.Options(show_trace = true)`: any other `Optim.Options` one wishes to pass to the optimizer

!!! note 
    * `optim_method`:
        * Only `Optim.ZerothOrderOptimizer`s and `Optim.FirstOrderOptimizer`s are possible for [`SchottkyAnoMaLy`](@ref). Higher-order methods require a Hessian which is frankly too expensive to compute exactly for any reasonable-sized [`InterpolationSet`](@ref).
        * The zeroth-order `Optim.NelderMead()` method seems to fare well in my tests, and may work well for large [`InterpolationSet`](@ref)s, but it *may* be a bit sketchy for constrained problems. So far, I haven't run into problems though.
    
    * `optim_algorithm`:
        * For constrained optimization, `Optim` suggests one use the `Fminbox` and so that is the only one I've tried.

## [`DonutVolcanoEnsemble`](@ref) Options

To be used in the [`InterpolationSet`](@ref) and the [`TrainingSet`](@ref) when constructing 
random ensembles for the forwards-problem.

* `max_num_dve = 30`: the maximum number of individual [`DonutVolcanoEnsemble`](@ref)s used to build a single random distribution
* `dve_mode_range = (0, 5)`: the rangle over which to uniformly generate peak positions for the [`DonutVolcanoEnsemble`](@ref)s
* `dve_width_range = dve_mode_range`: the range over which to uniformly generate widths for the [`DonutVolcanoEnsemble`](@ref)s

!!! note
    If `dve_width_range[1]` is too small compared to the characteristic temperature scale,
    the corresponding sharp peaks cannot be resolved, as they seem to contribute minimally 
    to the specific heat.

## Chebyshev Polynomial Interpolation Options

These are used to obtain the interpolated, trained, and learned Chebyshev coefficients.

* `cheby_order = 300`: the order of the Chebyshev polynomial used for the interpolation of the Schottky barrier distribution
* `distribution_domain = (0, 30)`: the interpolation interval for the Schottky barrier distributions

## Integration Options 

These are for `QuadGK.quadgk` and `NumericalIntegration.integrate`.

* `quadgk_rtol = sqrt(eps(T))`: the convergence relative tolerance in the adaptive Gauss Kronod integration to obtain the specific heat at a given temperature
* `quadgk_domain = (0, Inf)`: the domain over which to integrate the ensembles to determine the specific heats
* `numint_method = TrapezoidalFast()`: the method used to integrate the difference between the specific heats (see [`msqdiff`](@ref) and friends) 

"""
struct SchottkyOptions{T <: AbstractFloat}
    # Default attributes
    # ---------------------------------------

    # ⇒ Analysis-wide parameters
    analysis_seed::Int
    analysis_rng::AbstractRNG
    analysis_nls::NLevelSystem
    analysis_iterations::Int

    # ⇒ Learning algorithm options
    num_interp::Int
    num_train::Int
    max_cheby_component::Int
    initial_hyperparameters::_SA_input_pair{T}
    #   ⇒ Optimization options
    optim_lb::_SA_input_pair{T}
    optim_ub::_SA_input_pair{T}
    optim_method::Union{Optim.ZerothOrderOptimizer, Optim.FirstOrderOptimizer}
    optim_algorithm::Optim.AbstractConstrainedOptimizer
    optim_options::Optim.Options

    # ⇒ DonutVolcanoEnsemble options for interpolation and training
    max_num_dve::Int
    dve_mode_range::_SA_input_pair{T}
    dve_width_range::_SA_input_pair{T}

    # ⇒ Chebyshev interpolation parameters for the learned distribution
    cheby_order::Int
    distribution_domain::_SA_input_pair{T}

    # ⇒ Integration parameters 
    quadgk_rtol::T
    quadgk_domain::_SA_input_pair{T}
    numint_method::NumericalIntegration.IntegrationMethod
end

function SchottkyOptions(;
                          analysis_type::Union{Type, Nothing} = nothing, # used to specify a different type
                          analysis_seed = 42,
                          analysis_rng = Xoshiro(analysis_seed),
                          analysis_nls = TwoLevelSystem(),
                          analysis_iterations = 1,
                          num_interp = 128,
                          num_train = num_interp,
                          max_cheby_component = 10,
                          initial_hyperparameters = (0.5, 0.5),
                          optim_lb = (1e-12, 1e-8),
                          optim_ub = (Inf, Inf),
                          optim_method = Optim.LBFGS(),
                          optim_algorithm = Optim.Fminbox(optim_method),
                          optim_options = Optim.Options(show_trace = true),
                          max_num_dve = 30,
                          dve_mode_range = nothing,
                          global_dve_mode_range = (0, 5),                 # Use these, appropriately scaled to global_distribution_domain, if dve_mode_range isa Nothing
                          dve_width_range = nothing,
                          global_dve_width_range = global_dve_mode_range, # Use these, appropriately scaled to global_distribution_domain, if dve_mode_range isa Nothing
                          cheby_order = 300,
                          distribution_domain, # No default is resonable! (It depends on temperature scale.)
                          global_distribution_domain = (0, 30),    # Use this to scale the dve_*_range when they are not supplied
                          quadgk_rtol = nothing, # Will be made into sqrt(eps()) in the function body
                          quadgk_domain = nothing,
                          global_quadgk_domain = (0, Inf),
                          numint_method = NumericalIntegration.TrapezoidalFast()
    )

    # Set and check the Chebyshev interpolation parameters
    # These need to be checked first because the control the scales and types of the others
    _cheby_order = cheby_order
    @assert _cheby_order ≥ 0 "The order of the Chebyshev interpolation must be a nonnegative number. Got $_cheby_order."
    _distribution_domain = distribution_domain
    Ttype = analysis_type isa Nothing ? eltype(_distribution_domain) : analysis_type
    @assert Ttype <: AbstractFloat "The elements of the supplied distribution_domain must be of AbstractFloat type. Got $Ttype."

    # Set and check the analysis-wide parameters
    _analysis_seed = analysis_seed
    _analysis_rng = (analysis_rng |> typeof)(_analysis_seed) # to be sure the seed matters...
    _analysis_nls = analysis_nls
    _analysis_iterations = analysis_iterations
    @assert _analysis_iterations ≥ 0 "The number of analysis_iterations ≥ 0. Got $_analysis_iterations."

    # Set and check the learning algorithm parameters
    _num_interp = num_interp
    @assert _num_interp > 1 "The size of the InterpolationSet must be greater than one. Got $_num_interp."
    _num_train = num_train
    @assert _num_train > 0 "The size of the TrainingSet must be greater than zero. Got $_num_train."
    _max_cheby_component = max_cheby_component
    @assert  0 ≤ _max_cheby_component ≤ _cheby_order "The maximum Chebyshev component to include in the loss must be greater than or equal to zero and less than the maximum index. Got $_max_cheby_component while the maximum is $_cheby_order."
    _initial_hyperparameters = Ttype.(initial_hyperparameters)

    # Set and check the Optim options
    _optim_lb = Ttype.(optim_lb)
    @assert all(_optim_lb .> zero(Ttype)) "The lower bounds for the constrained optimization must all be greater than zero. Got $_optim_lb."
    _optim_ub = Ttype.(optim_ub)
    @assert all(_optim_ub .> _optim_lb) "The upper bounds for the constrained optimization must all be greater than the lower bounds. Got $_optim_ub while the lower bounds are $_optim_lb."
    _optim_method = optim_method
    @assert _optim_method isa Union{Optim.ZerothOrderOptimizer, Optim.FirstOrderOptimizer} "The supported optimization methods must be either a Optim.ZerothOrderOptimizer or a Optim.FirstOrderOptimizer. Got $_optim_method."
    _optim_algorithm = optim_algorithm
    @assert _optim_algorithm isa Optim.AbstractConstrainedOptimizer "The only supported optimization algorithms are constrained. Got $_optim_algorithm."
    _optim_options = optim_options
    @assert _optim_options isa Optim.Options "The optim_options keyword argument is to provide Optim.Options to the optimization routines. Please supply an instance of that type (or use the default)."

    # Set and check the DonutVolcanoEnsemble parameters
    _max_num_dve = max_num_dve
    @assert _max_num_dve > 0 "The maximum number of DonutVolcanoEnsembles in a single random distribution of barriers must be greater than zero."
    _dve_mode_range = dve_mode_range
    if _dve_mode_range isa Nothing
        Δext = _distribution_domain[end] - _distribution_domain[begin]
        gΔext = global_distribution_domain[end] - global_distribution_domain[begin]
        _dve_mode_range = Δext / gΔext .* global_dve_mode_range
    end
    _dve_mode_range = Ttype.(_dve_mode_range)
    @assert _distribution_domain[begin] ≤ _dve_mode_range[begin] && _dve_mode_range[end] ≤ _distribution_domain[end] "The supplied mode range must lie within the distribution domain. Got $_dve_mode_range while the domain is $_distribution_domain."
    _dve_width_range = dve_width_range
    if _dve_width_range isa Nothing
        Δext = _distribution_domain[end] - _distribution_domain[begin]
        gΔext = global_distribution_domain[end] - global_distribution_domain[begin]
        _dve_width_range = Δext / gΔext .* global_dve_width_range
    end
    _dve_width_range = Ttype.(_dve_width_range)
    @assert _distribution_domain[begin] ≤ _dve_width_range[begin] && _dve_width_range[end] ≤ _distribution_domain[end] "The supplied width range must lie within the distribution domain. Got $_dve_width_range while the domain is $_distribution_domain."

    # Set and check the various integration parameters
    _quadgk_rtol = quadgk_rtol isa Nothing ? sqrt(eps(Ttype)) : Ttype(quadgk_rtol)
    @assert _quadgk_rtol > zero(Ttype) "The relative tolerance for the Gauss-Kronod adaptive quadrature must be greater than zero. Got $_quadgk_rtol."
    _quadgk_domain = quadgk_domain isa Nothing ? Ttype.(global_quadgk_domain) : Ttype.(quadgk_domain)
    if !all( _quadgk_domain .== Ttype.(global_quadgk_domain) ) 
        @warn "The domain for the Gauss-Kronod adaptive quadrature is set to something other than (0, Inf). Was this a mistake?" quadgk_domain global_quadgk_domain
    end
    _numint_method = numint_method
    @assert _numint_method isa NumericalIntegration.IntegrationMethod "The numint_method keyword must specify a NumericalIntegration.IntegrationMethod. Got $_numint_method."

    return SchottkyOptions{Ttype}(
        _analysis_seed, _analysis_rng, _analysis_nls, _analysis_iterations,
        _num_interp, _num_train, _max_cheby_component, _initial_hyperparameters,
        _optim_lb, _optim_ub, _optim_method, _optim_algorithm, _optim_options,
        _max_num_dve, _dve_mode_range, _dve_width_range,
        _cheby_order, _distribution_domain, 
        _quadgk_rtol, _quadgk_domain, _numint_method
    )
end

Base.eltype(::SchottkyOptions{T}) where T = T

"""
    SchottkyAnalysis{T <: AbstractFloat}

A `struct` to contain the data for the machine learning analysis.

# Contents

This container is split into input and analysis-generated data.

## Input data

The user need only specify the following:

* `temperatures::Vector{T}`: the list of temperatures 
* `input_cV::Vector{T}`: the corresponding specific heat values to fit 
* `opts::SchottkyOptions{T}`: analysis options

## Analysis-generated data

These values will be calculated automatically either on initialization,
during training, or while making predictions.

* `predictors::Vector{EnsemblePredictor{T}}`: the collection of [`EnsemblePredictor`](@ref) for each analysis iteration
* `predictions::Vector{Vector{T}}`: the predicted Chebyshev coefficients for each analysis iteration
"""
struct SchottkyAnalysis{T <: AbstractFloat}
    # Input data
    temperatures::Vector{T}
    input_cV::Vector{T}
    opts::SchottkyOptions{T}

    # Random number stuff for reproducibility
    rdveg::RandomDonutVolcanoGenerator{T}

    # EnsemblePredictors
    predictors::Vector{EnsemblePredictor{T}}

    # Predictions
    predictions::Vector{Vector{T}}
end

function SchottkyAnalysis(temps, input_cV, opts::SchottkyOptions; initial_prediction::Bool = false)
    @assert length(temps) == length(input_cV) "Size mismatch. There must be as many temperatures as specific heats. Got $(length(temps)) and $(length(input_cV))."

    Temp_type = eltype(temps)
    rdveg = RandomDonutVolcanoGenerator{Temp_type}(opts.max_num_dve, opts.dve_mode_range[end], opts.dve_width_range[end])
    predictors = EnsemblePredictor{Temp_type}[]
    predictions = Vector{Temp_type}[]
    for _ ∈ UnitRange(1, opts.analysis_iterations)
        push!( predictors,
               EnsemblePredictor(
                opts.analysis_rng,
                rdveg,
                opts.num_interp,
                opts.num_train,
                temps,
                opts.cheby_order,
                opts.distribution_domain,
                opts.initial_hyperparameters,
                opts.max_cheby_component,
                opts.quadgk_domain[begin],
                opts.quadgk_domain[end],
                opts.analysis_nls,
                opts.quadgk_rtol,
                opts.numint_method
               )
        )
        push!( predictions, zeros(Temp_type, opts.cheby_order) )
    end

    sa = SchottkyAnalysis{Temp_type}( temps, input_cV, opts, rdveg, predictors, predictions )
    initial_prediction ? predict!(sa) : nothing

    return sa
end

train!(sa::SchottkyAnalysis, predictor_idx) = train!(sa.predictors[predictor_idx], sa.opts)
function train!(sa::SchottkyAnalysis)
    for idx ∈ eachindex(sa.predictors)
        train!(sa, idx)
    end
    return sa
end

function predict!(trained_sa::SchottkyAnalysis, predictor_idx; combine_sets = false)
    trainσ = hyperparameters(trained_sa.predictors[predictor_idx].gkrr)[begin]
    pred_coeffs = view(trained_sa.predictions[predictor_idx], :)
    predictor = trained_sa.predictors[predictor_idx]
    if combine_sets
        predictor = combine_to_predict(trained_sa, predictor_idx)
    end
    return predict!(pred_coeffs, trained_sa.temperatures, trained_sa.input_cV, 
                    predictor, trainσ, trained_sa.opts.numint_method)
end
function predict!(sa::SchottkyAnalysis; kwargs...)
    @show kws = Dict(kwargs...)
    if haskey(kws, :combine_sets) && kws[:combine_sets]
        @info "Combining Interpolation and Training Sets for each analysis_iteration."
    end
    for idx ∈ eachindex(sa.predictors)
        predict!(sa, idx; kwargs...)
    end
    return sa
end

create_chebinterp(predicted_sa::SchottkyAnalysis, pred_idx) = create_chebinterp(predicted_sa.predictions[pred_idx], predicted_sa.opts.distribution_domain...)
function create_chebinterp(predicted_sa::SchottkyAnalysis)
    output = []
    for pred_idx ∈ eachindex(predicted_sa.predictions)
        push!(output, create_chebinterp(predicted_sa, pred_idx))
    end
    return output
end

save_analysis(filename, sa::SchottkyAnalysis) = JLD2.save_object(filename, sa)
load_analysis(filename) = JLD2.load_object(filename)::SchottkyAnalysis