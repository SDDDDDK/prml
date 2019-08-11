module metrics
    export binary_cross_entropy_error, mean_squared_error, root_mean_squared_error, r2_score

    function binary_cross_entropy_error( y, t ; size_average=true )
        N = size(y, 1)
        E = - sum( t .* log.(y) .+ (1 .- t) .* log.(1 .- y) )
        if size_average return E / N
            else return E
        end
    end

    function mean_squared_error( y, t )
        sum( (y .- t).^2 ) / size(y, 1)
    end

    function root_mean_squared_error( y, t )
        sqrt( mean_squared_error( y, t ) )
    end

    function r2_score( y, t )
        1 - sum( (t .- y).^2 ) / sum( (t .- sum(t) / size(t, 1) ).^2 )
    end
    
    function accuracy( y, t )
        N = size(y, 1)
        sum(y .== t) / N
    end
end