using Plots
using Random
using Statistics

include("../../functions/utils.jl")
using .utils

Random.seed!(3)

N = 50
red = randn(N, 2) .+ [5.5 -1.0]
blue = randn(N, 2) .+ [2.0 2.5]

X = vcat(red, blue)
t = vcat(ones(N), ones(N).*-1)

xline = collect(range(-1, 9, length=10000))


N = size(X, 1)
X = homogeneous_coordinates( X )
w = randn( size( X, 2 ) )
history_w = [copy(w)]

no_update_iter = 0
max_iter = N^2

iter = 0;
while no_update_iter < N
    idx = rem(iter, N) + 1
    x = reshape( X[idx, :], 1, : )
    y = sign.(x * w)

    if y[1] != t[idx]
        global no_update_iter = 0
        w .+= t[idx] * x[:]
        push!( history_w, copy(w) )
    else
        global no_update_iter += 1
    end

    global iter += 1
    if iter > max_iter
        println("This dataset is not linear separable.")
        break
    end
end


anim = @animate for i=1:size(history_w, 1)
    w = history_w[i]
    yline = ( - w[2] * xline .- w[1] ) ./ w[3]

    plot(xline, yline, label="predicted dicision boundary")
    scatter!(red[:, 1], red[:, 2], label="red", color="red")
    scatter!(blue[:, 1], blue[:, 2], label="blue", color="blue")
    xlims!(-1, 9)
    ylims!(-4, 6)
end

mp4(anim, "perceptron_animation.mp4", fps=1);