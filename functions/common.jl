module common
    export sigmoid
    function sigmoid( x; α=1.0 )
        1 ./ (1 .+ exp.(-α.*x))
    end
end
