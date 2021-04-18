module SimpleDataLearner

using LinearAlgebra
using ForwardDiff

import Base.map
 
# Basic differential operators we need:
∇(f, x) = ForwardDiff.gradient(f, x)

abstract type Model end
mutable struct AffineModel <: Model
    W; b
end

struct Observation
    X; Y
end

map(model :: AffineModel, X) = model.W * X + model.b

"""
For the moment, perform a gradient step with respect to the quadratic loss. We generalize later.
"""
function gradientStep!(model :: AffineModel, o :: Observation, η :: Float64) :: AffineModel
    α = map(model, o.X) - o.Y
    model.W = model.W - 2*η*o.X*α
    model.b = model.b - 2*η*α
end

function fit!(model :: Model, observations :: Array{Observation})
end

end # module