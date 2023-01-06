
export EnsemblePredictor

"""
    EnsemblePredictor{T <: AbstractFloat}

Container around the individual [`DonutVolcanoEnsemble`](@ref)s used 
for the forwards-problem in both the [`InterpolationSet`](@ref) and 
the [`TrainingSet`](@ref). The latter two `Set`s are contained within 
a wrapped [`GaussianKRRML`](@ref) along with its gradient [`∇GaussianKRRML`](@ref).

This object, in effect, contains all information to "solve" for the 
Schottky barrier distribution, for *any* given temperatures and 
input specific heat values.
"""
struct EnsemblePredictor{T <: AbstractFloat}
    interp_dve::Vector{DonutVolcanoEnsemble{T}}
    train_dve::Vector{DonutVolcanoEnsemble{T}}

    gkrr::GaussianKRRML{T}
    gradgkrr::∇GaussianKRRML{T}
end

