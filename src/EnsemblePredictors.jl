
using Random 
using LinearAlgebra

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

function EnsemblePredictor(rng::AbstractRNG, 
                           rdveg::RandomDonutVolcanoGenerator, 
                           interp_size, 
                           train_size, 
                           temps, 
                           cheby_order,
                           interpdomain,
                           initial_hyp,
                           max_loss_order,
                           Δmin,
                           Δmax,
                           nls,
                           quadgk_rtol,
                           num_int_method)
    Temp_type = eltype(temps)
    # Generate ensembles
    interp_ens = rand(rng, rdveg, interp_size)
    train_ens = rand(rng, rdveg, train_size)
    # Compute their specific heats 
    interp_cVs = compute_cVs(interp_ens, temps; nls = nls, Δmin = Δmin, Δmax = Δmax, rtol = quadgk_rtol )
    train_cVs = compute_cVs(train_ens, temps; nls = nls, Δmin = Δmin, Δmax = Δmax, rtol = quadgk_rtol )
    # Focus on interp stuff
    interp_msqdiff_mat = populate_msqdiffs(interp_cVs, temps; method = num_int_method )    
    interp_chebys = chebycoefficients(interp_ens, cheby_order, interpdomain...) # This has a default argument coefftol = zero which will likely not change
    # Turn to the train stuff
    train_chebys = chebycoefficients(train_ens, cheby_order, interpdomain...) |> transpose
    # Create the GaussianKRRML learning module
    gkrr = GaussianKRRML(temps, interp_cVs, interp_chebys, interp_msqdiff_mat, train_cVs, train_chebys; σ0 = initial_hyp[1], λ0 = initial_hyp[2], max_loss_order = max_loss_order)
    ∂gkrr = ∇GaussianKRRML(gkrr)
    # Return the proper EnsemblePredictor
    return EnsemblePredictor{Temp_type}( interp_ens, train_ens, gkrr, ∂gkrr )
end

