module SimpleDataLearner
import Base.map
using LinearAlgebra
mutable struct Approximation
    W; b
end

map(f :: Approximation, X) = f.W * X + f.b
end # module
