"""MLP in Flux.jl to add the inputs"""

using Flux

# Training data
npoints = 10
ndim = 2
xtrain = rand(npoints, ndim)  # Python convention
ytrain = sum(xtrain, dims=2)  # this feels better that `ytrain = sum(eachcol(xtrain))`

# Model definition and constructors
struct Adder
    """No bias for this lowly adder, just weights"""
    W
end
Adder(ndim::Integer) = Adder(randn(ndim, 1))

# Implement forward pass
function forward(m::Adder, x)
    y = x * m.W
    return y
end
(m::Adder)(x) = forward(m, x)

# Instantiate the model
model = Adder(2)

# Check forward pass
println("Before training:")
println("Adder says 1 + 1 = $(first(model([1 1])))")

# Loss function
function mse(ypred, ytrue)
    sqerror = sum((ypred .- ytrue).^2, dims=1)
    mse = (1.0 / length(ytrue)) * sqerror
    return sum(mse)
end

# Optimizer
optimizer = ADAM(0.1)

# Zygote
weights = params(model.W)

# Training loop
for i in 1:500
    gradients = gradient(() -> mse(model(xtrain), ytrain), weights)
    Flux.Optimise.update!(optimizer, model.W, gradients[model.W])
    loss = mse(model(xtrain), ytrain)
    if i % 50 == 0
        println("Epoch: $i, Loss: $loss")
    end
end

# Let's see if we can now add
println("After training:")
println("Adder says 1 + 1 = $(first(model([1 1])))")
println("Adder says 1.2 + 0.8 = $(first(model([1.2 0.8])))")
println("Adder says 1e2 + 1e3 = $(first(model([1e2 1e3])))")

# Examine model weights
println(model.W)