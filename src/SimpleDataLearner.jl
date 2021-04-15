module SimpleDataLearner
import Base.map
using LinearAlgebra

mutable struct AffineApproximation
    W; b
end
struct Datum
    X; Y
end

map(f :: AffineApproximation, X) = f.W * X + f.b

function learn!(f :: AffineApproximation, data :: Array{Datum})
end

end # module