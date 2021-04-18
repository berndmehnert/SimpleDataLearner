module SimpleDataLearner

import Base.map
using LinearAlgebra
using ForwardDiff
 
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

function quadraticLoss(model :: Model, o :: Observation) 
    α = map(model, o.X) - o.Y
    return dot(α, α)
end

function gradientStep!(model :: AffineModel, o :: Observation, η :: Float64) :: AffineModel
    α = quadraticLoss(model, o)
    model.W = model.W - 2*η*o.X*α
    model.b = model.b - 2*η*α
end

function learn!(model :: Model, loss, data :: Array{Observation})
end

end # module