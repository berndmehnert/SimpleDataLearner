module SimpleDataLearner

using LinearAlgebra
using ForwardDiff

export AffineTransformation, ActivationFunction
 
# Basic differential operators we need:
∇(f, x) = ForwardDiff.gradient(f, x)

abstract type AbstractModel end
abstract type AbstractTransformation end
mutable struct AffineTransformation <: AbstractTransformation
    W :: Matrix{Float64}
    b :: Matrix{Float64}
end
mutable struct Convolution <: AbstractTransformation
end

@enum ActivationFunction begin
    RELU 
    GELU 
    Softmax
end

ModelComponent = Union{AbstractTransformation, ActivationFunction}

struct Model <: AbstractModel 
    components :: Vector{ModelComponent}
end

# Compute derivatives of transformations and activation functions
D(transformation :: AffineTransformation, X) = transformation.W 
function D(activationFunction :: ActivationFunction, X)
    if activationFunction == RELU 
        return (x -> x > 0).(X)
    elseif activationFunction == Softmax
        return exp.(X)/sum(exp.(X))
    else 
        error("Activation function not available ..")
    end
end

apply(transformation :: AffineTransformation, X) = transformation.W * X + transformation.b

function getActivationFunction(activationFunction :: ActivationFunction)
    if activationFunction == RELU 
        return X -> max.(X,0)
    elseif activationFunction == Softmax
        return X -> exp.(X)/sum(exp.(X))
    else 
        error("Activation function not available ..")
    end
end

end # module