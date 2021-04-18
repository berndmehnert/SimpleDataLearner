module SimpleDataLearner

import Base.map
using LinearAlgebra
using ForwardDiff

""" 
Basic differential operators we need:
"""
∇(f, x) = ForwardDiff.gradient(f, x)

abstract type Model end
mutable struct AffineModel <: Model
    W; b
end

struct Observation
    X; Y
end

map(model :: AffineModel, X) = model.W * X + model.b

function QuadraticLoss(model :: Model, o :: Observation) 
    α = map(model, o.X) - o.Y
    return dot(α, α)
end 

function learn!(model :: Model, loss, data :: Array{Observation})
end

end # moduleIsrael Keyes