module neural_networks
    module neural_networks_layer
        export LinearLayer, SigmoidLayer, BinaryCrossEntropyLayer, forward, backward

        using Random: randn

        include("../functions/common.jl")
        include("../functions/design_matrix.jl")
        include("../functions/metrics.jl")
        using .common: sigmoid
        using .design_matrix: homogeneous_coordinates
        using .metrics: binary_cross_entropy_error

        abstract type AbstractNeuralNetworksLayer end
        abstract type AbstractNeuralNetworksTrainableLayer <: AbstractNeuralNetworksLayer end
        abstract type AbstractNeuralNetworksUntrainableLayer <: AbstractNeuralNetworksLayer end

        mutable struct LinearLayer <: AbstractNeuralNetworksTrainableLayer
            w::Array{Real, 2}
            x::Array{Real, 2}
            ∇w::Array{Real, 2}
            function LinearLayer( in_features, out_features )
                w = randn( in_features+1, out_features )
                x = zeros( 1, 1 )
                ∇w = zeros( size(w) )
                new( w, x, ∇w )
            end
        end

        function forward( layer::LinearLayer, x )
            layer.x = homogeneous_coordinates( x );
            layer.x * layer.w
        end

        function backward( layer::LinearLayer, δ )
            layer.∇w = layer.x' * δ
            δ * layer.w[2:end, :]'
        end

        mutable struct SigmoidLayer <: AbstractNeuralNetworksUntrainableLayer
            z::Array{Real, 2}
            function SigmoidLayer( )
                z = zeros(1, 1)
                new( z )
            end
        end

        function forward( layer::SigmoidLayer, x )
            layer.z = sigmoid( x )
        end

        function backward( layer::SigmoidLayer, δ )
            δ .* layer.z .* ( 1 .- layer.z)
        end

        mutable struct BinaryCrossEntropyLayer <: AbstractNeuralNetworksUntrainableLayer
            y::Array{Real, 1}
            t::Array{Real, 1}
            function BinaryCrossEntropyLayer( )
                y = []
                t = []
                new( y, t )
            end
        end

        function forward( layer::BinaryCrossEntropyLayer, y, t )
            layer.y = reshape( y, size(y, 1))
            layer.t = reshape( t, size(t, 1))
            binary_cross_entropy_error( layer.y, layer.t )
        end

        function backward( layer::BinaryCrossEntropyLayer )
            δ = zeros( size(layer.y) )
            δ[ layer.t .== 0 ] .= (1 ./ (1 .- layer.y))[ layer.t .== 0 ]
            δ[ layer.t .== 1 ] .= (-1 ./ layer.y)[ layer.t .== 1 ]
            δ
        end
    end

    module neural_networks_optimizer
        export StochasticGradientDescent, update
        import ..neural_networks_layer: AbstractNeuralNetworksLayer, AbstractNeuralNetworksTrainableLayer

        abstract type AbstractNeuralNetworksOptimizer end

        mutable struct StochasticGradientDescent <: AbstractNeuralNetworksOptimizer
            trainable_layers::Array{AbstractNeuralNetworksTrainableLayer, 1}
            η::Real
            function StochasticGradientDescent( layers ; η=1e-3 ) # ::Array{AbstractNeuralNetworksLayer, 1}
                trainable_layers = Array{AbstractNeuralNetworksTrainableLayer, 1}()
                for layer=layers
                    if isa(layer, AbstractNeuralNetworksTrainableLayer)
                        push!(trainable_layers, layer)
                    end
                end
                new( trainable_layers, η )
            end
        end

        function update( optimizer::StochasticGradientDescent )
            for layer=optimizer.trainable_layers
                layer.w .-= optimizer.η .* layer.∇w
            end
        end
    end
end