module linear_regression
    export LinearRegression, RidgeRegression, fit, predict, score

    using LinearAlgebra: I
    include("../functions/design_matrix.jl")
    include("../functions/metrics.jl")
    using .design_matrix: homogeneous_coordinates
    using .metrics: r2_score
    
    abstract type AbstractLinearRegression end

    mutable struct LinearRegression <: AbstractLinearRegression
        w::Array{Real, 1}
        function LinearRegression()
            w = []
            new(w)
        end
    end

    function fit( model::LinearRegression, X, t )
        model.w = inv(X' * X) * X' * t;
    end

    mutable struct RidgeRegression <: AbstractLinearRegression
        w::Array{Real, 1}
        λ::Real
        function RidgeRegression( λ::Real )
            w = []
            new(w, λ)
        end
    end

    function fit( model::RidgeRegression, X, t )
        model.w = inv(X' * X + model.λ * I) * X' * t;
    end

    function predict( model::AbstractLinearRegression, X )
        X * model.w
    end

    function score( model::AbstractLinearRegression, X, t )
        y = predict( model, X )
        r2_score( y, t )
    end
end