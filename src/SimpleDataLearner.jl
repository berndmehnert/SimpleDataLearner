module SimpleDataLearner

using LinearAlgebra
using ForwardDiff
import Base.map

export AffineTransformation, ActivationFunction
 
# Basic differential operators we need:
∇(f, x) = ForwardDiff.gradient(f, x)

abstract type AbstractModel end
abstract type AbstractTransformation end
mutable struct AffineTransformation <: AbstractTransformation
    W :: Matrix{Float64}
    b :: Vector{Float64}
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

map(transformation :: AffineTransformation, v) = transformation.W * v + transformation.b

function getActivationFunction(activationFunction :: ActivationFunction) :: Function
    if activationFunction == RELU 
        return v -> max.(v,0)
    elseif activationFunction == Softmax
        return v -> exp.(v)/sum(exp.(v))
    else 
        error("Activation function not available ..")
    end
end

map(activationFunction :: ActivationFunction, v) = getActivationFunction(activationFunction)(v)

map(model :: Model, v) = begin
    if length(model.components) > 1
        model1 = Model(model.components[1:end-1])
        w = map(model1, v)
        return map(model.components[end], w)
    else if length(model.components) == 1
        return map(model.components[1], v)
    else
        return nothing        
end

end # module