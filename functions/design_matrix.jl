module design_matrix
    export homogeneous_coordinates, polynomial_design_matrix, gauss_design_matrix

    function homogeneous_coordinates( X )
        if ndims(X) == 1 return vcat(ones(1), X)
        else return hcat(ones(size(X, 1)), X)
        end
    end

    function polynomial_design_matrix( x, M )
        N = size(x, 1)
        Φ = zeros(N, M+1)
        for j=0:M
            Φ[:, j+1] = x .^ j
        end
        Φ
    end

    function gauss_design_matrix( x, M, μ_range, s )
        N = size(x, 1)
        Φ = zeros(N, M+1)
        μs = collect(range(μ_range[1], μ_range[2],length=M+1))
        for j=0:M
            Φ[:, j+1] = exp.( (x .- μs[j+1]).^2 ./ (2s^2) )
        end
        Φ
    end
end