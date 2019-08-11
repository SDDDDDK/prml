module neural_networks_optimizer
    export StochasticGradientDescent

    using Random: randn

    include("../functions/common.jl")
    include("../functions/metrics.jl")
    include("../functions/utils.jl")
    using .common: sigmoid
    using .metrics: binary_cross_entropy_error
    using .utils: homogeneous_coordinates

    include("./neural_networks_layer.jl")
    using .neural_networks_layer: AbstractNeuralNetworksLayer, AbstractNeuralNetworksTrainableLayer

    abstract type AbstractNeuralNetworksOptimizer end

    mutable struct StochasticGradientDescent <: AbstractNeuralNetworksOptimizer
        trainable_layers::Array{AbstractNeuralNetworksTrainableLayer, 1}
        η::Real
        function StochasticGradientDescent( layers ; η=1e-3 ) # ::Array{AbstractNeuralNetworksLayer, 1}
            println(typeof(layers))
            trainable_layers = Array{AbstractNeuralNetworksTrainableLayer, 1}()
            for layer=layers
                println(typeof(layer))
                if isa(layer, AbstractNeuralNetworksTrainableLayer)
                    push!(trainable_layers, layer)
                end
            end
            new( trainable_layers, η )
        end
    end

    function step( optimizer::StochasticGradientDescent )
        for layer=optimizer.trainable_layers
            layer.w .-= optimizer.η .* layer.∇w
        end
    end
end