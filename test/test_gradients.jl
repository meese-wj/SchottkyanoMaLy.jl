
using SchottkyAnoMaLy
using Test

@info "Testing Gradient of Loss function"
@time @testset "Gradient of Loss function" begin

    @testset "2x2 analytic comparison" begin
        _σ = rand()
        _λ = rand()
        _m12 = rand()
        _mi1 = rand()
        _mi2 = rand()
        _f1 = rand()
        _f2 = rand()
        
        # define constant vectors and matrices
        msqdiff_2vec = [_mi1, _mi2]
        msqdiff_2x2 = [[0 _m12]; 
                       [_m12 0]]
        fcomp_2vec = [_f1, _f2]

        # Compute kernel vectors and matrices
        νvec = gausskernel(msqdiff_2vec, _σ)
        ∂νvec = ∂σ_gausskernel(msqdiff_2vec, νvec, _σ)

        Kmat_2x2 = gausskernel(msqdiff_2x2, _σ)
        ∂Kmax_2x2 = ∂σ_gausskernel( msqdiff_2x2, Kmat_2x2, _σ )

        # Invert M = K + _λI from SchottkyAnoMaLy
        Minv = regularized_inverse( Kmat_2x2, _λ )

        # Use the SchottkyAnoMaLy formulas for the predictions and gradients
        # We take the gradient at (predicted, value) = (1, 0) to remove the loss contribution
        fpred = predict_component(νvec, Minv, fcomp_2vec)
        calc_grad = single_component_loss_gradient(1., 0., νvec, ∂νvec, ∂Kmax_2x2, Minv, fcomp_2vec)

        # Now build the analytic expressions for 2x2 case
        function simple_terms( msq12, msqvec, fvec )
            m12 = msq12
            mi1 = msqvec[1]
            mi2 = msqvec[2]
            f1 = fvec[1]
            f2 = fvec[2]
            return m12, mi1, mi2, f1, f2
        end

        # Calculate the analytic value of f
        function f_analytic_2x2( σ, λ, msq12, msqvec, fvec )
            m12, mi1, mi2, f1, f2 = simple_terms(msq12, msqvec, fvec)
            denom = -1 + exp( m12 / σ^2 ) * (1 + λ)^2
            first  =  -exp( (m12 - mi2)/(2σ^2) ) * f1 
            second =  -exp( (m12 - mi1)/(2σ^2) ) * f2
            third  = exp( -(-2m12 + mi1)/(2σ^2) ) * f1 * (1 + λ)
            fourth = exp( -(-2m12 + mi2)/(2σ^2) ) * f2 * (1 + λ)
            return (first + second + third + fourth) / denom
        end

        # Calculate the derivative of the predicted value f with respect to σ
        function ∂σf_analytic_2x2( σ, λ, msq12, msqvec, fvec )
            m12, mi1, mi2, f1, f2 = simple_terms(msq12, msqvec, fvec)
            denom = σ^3 * (-1 + exp( m12 / σ^2 ) * (1 + λ)^2)^2
            first   = exp( (m12 - mi1)/(2σ^2) ) * f2 * (m12 - mi1)
            second  = exp( (m12 - mi2)/(2σ^2) ) * f1 * (m12 - mi2)
            third   = -exp( -(-2m12 + mi1)/(2σ^2) ) * f1 * (2m12 - mi1) * (1 + λ)
            fourth  = -exp( -(-2m12 + mi2)/(2σ^2) ) * f2 * (2m12 - mi2) * (1 + λ)
            fifth   = exp( -(-3m12 + mi1)/(2σ^2) ) * f2 * (m12 + mi1) * (1 + λ)^2
            sixth   = exp( -(-3m12 + mi2)/(2σ^2) ) * f1 * (m12 + mi2) * (1 + λ)^2
            seventh = -exp( -(-4m12 + mi1)/(2σ^2) ) * f1 * mi1 * (1 + λ)^3
            eighth  = -exp( -(-4m12 + mi2)/(2σ^2) ) * f2 * mi2 * (1 + λ)^3
            return -( first + second + third + fourth + fifth + sixth + seventh + eighth ) / denom
        end

        # Calculate the derivative of the predicted value f with respect to λ
        function ∂λf_analytic_2x2( σ, λ, msq12, msqvec, fvec )
            m12, mi1, mi2, f1, f2 = simple_terms(msq12, msqvec, fvec)
            denom = (-1 + exp( m12 / σ^2 ) * (1 + λ)^2)^2
            prefactor = exp( -(-2m12 + mi1 + mi2)/(2σ^2) )
            first  = exp( mi2/(2σ^2) ) * f1
            second = exp( mi1/(2σ^2) ) * f2
            third  = -2exp( (m12 + mi1)/(2σ^2) ) * f1 * (1 + λ)
            fourth = -2exp( (m12 + mi2)/(2σ^2) ) * f2 * (1 + λ)
            fifth  = exp( (2m12 + mi2)/(2σ^2) ) * f1 * (1 + λ)^2
            sixth  = exp( (2m12 + mi1)/(2σ^2) ) * f2 * (1 + λ)^2
            return -prefactor * ( first + second + third + fourth + fifth + sixth ) / denom
        end

        # Combine partials into a gradient ∇ = (∂σ, ∂λ)
        ∇f_analytic_2x2( σ, λ, msq12, msqvec, fvec ) = (∂σf_analytic_2x2(σ, λ, msq12, msqvec, fvec), ∂λf_analytic_2x2(σ, λ, msq12, msqvec, fvec))

        # Calculate the analytic values
        fanalytic = f_analytic_2x2( _σ, _λ, msqdiff_2x2[1, 2], msqdiff_2vec, fcomp_2vec )
        gradfanalytic = ∇f_analytic_2x2(_σ, _λ, msqdiff_2x2[1, 2], msqdiff_2vec, fcomp_2vec)
        
        # Test that they are the same as what is used in SchottkyAnoMaLy
        @test fpred ≈ fanalytic
        @test ( calc_grad .≈ gradfanalytic) |> all
    end

end 
println()