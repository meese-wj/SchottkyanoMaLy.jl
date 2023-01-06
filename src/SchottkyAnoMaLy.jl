module SchottkyAnoMaLy

using Random
using Distributions
using FastChebInterp
using LinearAlgebra

include("PackageHelpers.jl")
include("StatisticalMechanicsFormulas.jl")
include("DonutVolcanoEnsembles.jl")
include("GaussianKernel.jl")
include("SpecificHeatLinearAlgebra.jl")
include("LearningFunctions.jl")
include("EuclideanKRR.jl")
include("EnsemblePredictors.jl")
include("SchottkyAnalysis.jl")

end # module SchottkyAnoMaLy
