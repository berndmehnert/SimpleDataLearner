module SimpleDataLearner

using LinearAlgebra
using ForwardDiff

export Dense, ∇, Softmax, RELU, Model, Jac, params
 
# Basic differential operators we need:
∇(f, x) = ForwardDiff.gradient(f, x)
Jac(f, x) = ForwardDiff.jacobian(f, x)

abstract type AbstractModel end
abstract type AbstractTransformation end
mutable struct AffineTransformation <: AbstractTransformation
    W :: Matrix{Float64}
    b :: Vector{Float64}
end

Dense(n, m) = AffineTransformation(rand(m, n), rand(m))

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

apply(transformation :: AffineTransformation, v) = transformation.W * v + transformation.b

function getActivationFunction(activationFunction :: ActivationFunction) :: Function
    if activationFunction == RELU 
        return v -> max.(v,0)
    elseif activationFunction == Softmax
        return v -> exp.(v)/sum(exp.(v))
    else 
        error("Activation function not available ..")
    end
end

apply(activationFunction :: ActivationFunction, v) = getActivationFunction(activationFunction)(v)

apply(model :: Model, v) = begin
    if length(model.components) > 1
        model1 = Model(model.components[1:end-1])
        w = apply(model1, v)
        return apply(model.components[end], w)
    elseif length(model.components) == 1
        return apply(model.components[1], v)
    else
        return nothing 
    end       
end

params(affineTransformation :: AffineTransformation) = [affineTransformation.b, affineTransformation.W]
params(activationFunction :: ActivationFunction) = []
params(model :: Model) = begin
    result = []
    if length(model.components) > 0
        for component in model.components
            append!(result, params(component))
        end
    end
    return result
end
end # module