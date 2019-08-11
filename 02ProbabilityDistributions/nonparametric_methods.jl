module nonparametric_methods
    export KernelDensityEstimation, KNearestNeiboursRegression, KNearestNeiboursClassification, fit, predict

    using Random: randn

    abstract type NonparametricMethods end

    mutable struct KernelDensityEstimation <: NonparametricMethods
        function KernelDensityEstimation(  )
            new()
        end
    end

    mutable struct KNearestNeiboursRegression <: NonparametricMethods
        k::Real
        X::Array{Real}
        t::Array{Real, 1}
        function KNearestNeiboursRegression( k )
            new(k)
        end
    end

    function fit( model::KNearestNeiboursRegression, X, t )
        model.X = X
        model.t = t
    end

    function predict( model::KNearestNeiboursRegression, X )
        if size(model.X, 2) != 1 && ndims(X) == 1
            X = reshape(X, 1, :)
        end

        n = size( X, 1 )
        y = zeros( n )

        for i=1:n
            x = X[i, :]
            if ndims(x) == 0
                distance = abs.(model.X .- x)
            else
                distance = sum( abs.(model.X .- reshape( x, 1, : ) ), dims=2)
            end
            top_k = sort( Pair.( distance, model.t ), dims=1 )[1:model.k]

            for top_i = top_k
                y[i] += top_i[2]
            end
            y[i] /= model.k
        end
        
        y
    end

    mutable struct KNearestNeiboursClassification <: NonparametricMethods
        k::Real
        X::Array{Real}
        t::Array{Real, 1}
        function KNearestNeiboursClassification( k )
            X = []
            new(k, X)
        end
    end

    function fit( model::KNearestNeiboursClassification, X, t )
        model.X = X
        model.t = t
    end

    function predict( model::KNearestNeiboursClassification, X )
        if size(model.X, 2) != 1 && ndims(X) == 1
            X = reshape(X, 1, :)
        end

        n = size(X, 1)
        y = zeros( n )

        for i=1:n
            x = X[i, :]
            if ndims(x) == 0
                distance = abs.(model.X .- x)
            else
                distance = sum( abs.(model.X .- reshape( x, 1, : ) ), dims=2)
            end
            top_k = sort(Pair.(distance, model.t), dims=1)[1:model.k]

            labels_in_k = Dict()
            for top_i = top_k
                label_i = top_i[2]
                if label_i in keys(labels_in_k)
                    labels_in_k[label_i] += 1
                else
                    push!( labels_in_k, label_i => 1 )
                end
            end
            
            y[i] = collect(keys(labels_in_k))[argmax(collect(values(labels_in_k)))]
        end

        y
    end
        
end