module SimpleDataLearner

using LinearAlgebra
using ForwardDiff
 
# Basic differential operators we need:
∇(f, x) = ForwardDiff.gradient(f, x)

abstract type AbstractModel end
abstract type AbstractTransformation end

mutable struct AffineTransformation <: AbstractTransformation
    W :: Matrix{Float64}
    b :: Matrix{Float64}
end

@enum ActivationFunction begin
    RELU
    Softmax
end

ModelComponent = Union{AbstractTransformation, ActivationFunction}

struct Model <: AbstractModel 
    components :: Vector{ModelComponent}
end

apply(transformation :: AffineTransformation, X :: Matrix{Float64}) = transformation.W * X + transformation.b

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