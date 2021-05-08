module SimpleDataLearner

using LinearAlgebra
using ForwardDiff

import Base.map
 
# Basic differential operators we need:
∇(f, x) = ForwardDiff.gradient(f, x)

abstract type AbstractModel end
mutable struct AffineModel <: AbstractModel
    W; b
end

struct RELU <: AbstractModel end
struct Softmax <: AbstractModel end
struct Model <: AbstractModel
    components :: Vector{AbstractModel}
end

struct Observation
    X; Y
end

map(model :: AffineModel, X) = model.W * X + model.b
map(model :: RELU, X) = (x -> max(x,0)).(X)
map(model :: Softmax, X) = x -> exp.(X)/sum(exp.(X))

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