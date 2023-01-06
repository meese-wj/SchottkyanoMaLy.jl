
using Optim

export SchottkyOptions

const _SA_input_pair = Tuple{T, T} where T # This is purely in anticipation of maybe adding Vector pair too for convenience

"""
    SchottkyOptions{T <: Real}

Keyword-constructed `struct` that handles all of the default 
configurable options for the SchottkyAnalysis.

# Options

These options are organized by purpose.

## Machine-Learning Algorithm Options

These are for the specific machine-learning algorithm that [`SchottkyAnoMaLy`](@ref)
implements.

* `num_learn = 128`: the number of ensembles in the [`InterpolationSet`](@ref)
* `num_train = num_interp`: the number of ensembles in the [`TrainingSet`](@ref)
* `max_cheby_component = 10`: the maximum component to consider in the [`total_loss`](@ref) when training the hyperparameters
* `initial_hyperparameters = (0.5, 0.5)`: the kickstart value for the `(σ, λ)` hyperparameters in [`GaussianKRRML`](@ref) model

### Optimization Options

These options are used by `Optim.jl` to tune the hyperparameters on the [`TrainingSet`](@ref).
Currently, we use a multivariate constrained optimization method since 

1. only the magnitude of the interpolation bandwidth `σ` matters,
2. the stiffness `λ > 0` to ensure boundedness of the solution and numerical stability.

* `optim_lb = (1e-12, 1e-8)`: the lower `(σ, λ)` bounds for the constrained optimization
* `optim_ub = (Inf, Inf):` the upper `(σ, λ)` bounds for the constrained optimization
* `optim_method = Optim.ConjugateGradient()`: the multivariate *unconstrained* optimization method
* `optim_algorithm = Optim.Fminbox(optim_method)`: the multivariate *constrained* optimization algorithm used on top of `optim_method`
* `optim_options = Optim.Options(show_trace = true)`: any other `Optim.Options` one wishes to pass to the optimizer

!!! note 
    * `optim_method`:
        * Only `Optim.ZerothOrderOptimizer`s and `Optim.FirstOrderOptimizer`s are possible for [`SchottkyAnoMaLy`](@ref). Higher-order methods require a Hessian which is frankly too expensive to compute exactly for any reasonable-sized [`InterpolationSet`](@ref).
        * The zeroth-order `Optim.NelderMead()` method seems to fare well in my tests, and may work well for large [`InterpolationSet`](@ref)s but it can be a bit sketchy for constrained problems.
    
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

## Analysis-wide Options

These parameters are for configuring the analysis globally.

* `analysis_seed = 42`: global random seed to set for reproducibility
* `analysis_nls = TwoLevelSystem()`: the type of statistical mechanics model used to form the specific heats from a barrier distribution

"""
Base.@kwdef struct SchottkyOptions{T <: Real}
    # Default attributes
    # ---------------------------------------

    # ⇒ Learning algorithm options
    num_interp::Int = 128
    num_train::Int = num_interp
    max_cheby_component::Int = 10
    initial_hyperparameters::_SA_input_pair{T} = T.((0.5, 0.5))
    #   ⇒ Optimization options
    optim_lb::_SA_input_pair{T} = T.((1e-12, 1e-8))
    optim_ub::_SA_input_pair{T} = T.((Inf, Inf))
    optim_method::Union{Optim.ZerothOrderOptimizer, Optim.FirstOrderOptimizer} = Optim.ConjugateGradient()
    optim_algorithm::Optim.AbstractConstrainedOptimizer = Fminbox(optim_method)
    optim_options::Optim.Options = Optim.Options(show_trace = true)

    # ⇒ DonutVolcanoEnsemble options for interpolation and training
    max_num_dve::Int = 30
    dve_mode_range::_SA_input_pair{T} = T.((0, 5))
    dve_width_range::_SA_input_pair{T} = dve_mode_range

    # ⇒ Chebyshev interpolation parameters for the learned distribution
    cheby_order::Int = 300
    distribution_domain::_SA_input_pair{T} = T.((0, 30))

    # ⇒ Analysis-wide parameters
    analysis_seed::Int = 42
    analysis_nls::NLevelSystem = TwoLevelSystem()

end