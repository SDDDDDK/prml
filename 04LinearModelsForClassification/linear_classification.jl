module linear_classification
    export FishersDiscriminant, Perceptron, LogisticRegression, fit, predict, predict_proba, score

    using Random: randn
    using Statistics: mean
    using LinearAlgebra: inv, norm, diagm

    include("../functions/common.jl")
    include("../functions/design_matrix.jl")
    include("../functions/metrics.jl")
    using .common: sigmoid
    using .design_matrix: homogeneous_coordinates
    using .metrics: accuracy

    abstract type AbstractLinearClassification end
    abstract type AbstractDiscriminantFunction <: AbstractLinearClassification end # 識別関数
    abstract type AbstractProbabilisticDiscriminativeModel <: AbstractLinearClassification end # 確率的識別モデル
    abstract type AbstractProbabilisticGenerativeModel <: AbstractLinearClassification end # 確率的生成モデル

    mutable struct FishersDiscriminant <: AbstractDiscriminantFunction
        w::Array{Real, 1}
        threshold::Real
        function FishersDiscriminant( )
            w = []
            threshold = 0.0
            new( w, threshold )
        end
    end
    
    function fit( model::FishersDiscriminant, X, t )
        X1 = X[t .== -1, :]
        X2 = X[t .== 1, :]
        
        mean1 = mean(X1, dims=1)
        mean2 = mean(X2, dims=1)
        
        mean_div = mean2 - mean1
        cov_between_class = mean_div' * mean_div
        
        x_mean_div1 = X1 .- mean1
        x_mean_div2 = X2 .- mean2
        cov_within_class = x_mean_div1' * x_mean_div1 + x_mean_div2' * x_mean_div2
        
        model.w = view( inv(cov_within_class) * mean_div', : )
        model.w = model.w ./ norm(model.w)
        
        max1 = maximum(X1 * model.w)
        min2  = minimum(X2 * model.w)
        if max1 < min2
            model.threshold = (max1 + min2) / 2
        else
            println("This dataset is not linear separable.")
        end
    end
    
    function predict( model::FishersDiscriminant, X )
        y = X * model.w
        y[ y .<  model.threshold ] .= -1
        y[ y .>= model.threshold ] .= 1
        y
    end

    mutable struct Perceptron <: AbstractDiscriminantFunction
        w::Array{Real, 1}
        eta::Real
        function Perceptron( ; eta=1.0 )
            w = []
            new(w, eta)
        end
    end

    function fit( model::Perceptron, X, t )
        N = size(X, 1)
        X = homogeneous_coordinates( X )
        model.w = randn( size( X, 2 ) )

        no_update_iter = 0
        max_iter = N^2

        iter = 0
        while no_update_iter < N
            idx = rem(iter, N) + 1
            x = reshape( X[idx, :], 1, : )
            y = predict( model, x[:, 2:end] )

            if y[1] != t[idx]
                no_update_iter = 0
                model.w .+= model.eta * t[idx] * x[:]
            else
                no_update_iter += 1
            end

            iter += 1
            if iter > max_iter
                println("This dataset is not linear separable.")
                break
            end
        end
    end

    function predict( model::Perceptron, X )
        X = homogeneous_coordinates( X )
        sign.( X * model.w ) 
    end

    mutable struct LogisticRegression <: AbstractProbabilisticDiscriminativeModel
        w::Array{Real, 1}
        max_iter::Integer
        function LogisticRegression( ; max_iter=100 )
            w = []
            new(w, max_iter)
        end
    end

    function fit( model::LogisticRegression, X, t )
        N = size( X, 1 )
        X = homogeneous_coordinates( X )
        model.w = randn( size( X, 2 ) )
        for iter=1:model.max_iter
            y = sigmoid( X * model.w )
            R = diagm( 0 => y .* (1 .- y) )
            try
                model.w .-= inv( X' * R * X ) * X' * (y - t)
            catch
                break
            end
        end
    end

    function predict( model::LogisticRegression, X )
        X = homogeneous_coordinates( X )
        y = sign.( X * model.w )
        y[ y .<= 0.0 ] .= 0.0
        y
    end

    function predict_proba( model::LogisticRegression, X )
        X = homogeneous_coordinates( X )
        sigmoid( X * model.w )
    end

    function score( model::AbstractLinearClassification, X, t )
        y = predict( X )
        accuracy( y, t )
    end
end
