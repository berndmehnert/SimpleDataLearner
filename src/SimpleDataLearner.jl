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

struct TrainingExample
    X; Y
end

map(model :: AffineModel, X) = model.W * X + model.b

function Loss(model :: Model, t :: TrainingExample) 
    α = map(model, t.X) - t.Y
    return α^2
end 

function learn!(model :: Model, loss, data :: Array{TrainingExample})
end

end # module
