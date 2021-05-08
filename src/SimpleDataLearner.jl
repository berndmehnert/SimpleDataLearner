module SimpleDataLearner

using LinearAlgebra
using ForwardDiff

import Base.map
 
# Basic differential operators we need:
∇(f, x) = ForwardDiff.gradient(f, x)

abstract type AbstractTransformation end
mutable struct AffineTransformation <: AbstractTransformation
    W; b
end

@enum ActivationFunction begin
    RELU
    Softmax
end

ModelComponent = Union{AbstractTransformation, ActivationFunction}
struct Model 
    components :: Vector{ModelComponent}
end

map(transformation :: AffineTransformation, X) = transformation.W * X + transformation.b
function get(activationFunction :: ActivationFunction)
    if activationFunction == RELU 
        return X -> max.(X,0)
    elseif activationFunction == Softmax
        return X -> exp.(X)/sum(exp.(X))
    else 
        error("Activation function not available ..")
    end
end

"""
For the moment, perform a gradient step with respect to the quadratic loss. We generalize later.
"""
function gradientStep!(model :: AffineModel, o :: Observation, η :: Float64) :: AffineModel
    α = map(model, o.X) - o.Y
    model.W = model.W - 2*η*o.X*α
    model.b = model.b - 2*η*α
end

function fit!(model :: Model, observations :: Array{Observation})
    η = 0.01
    for o in observations
        gradientStep!(model, o, η)
    end
end

end # module