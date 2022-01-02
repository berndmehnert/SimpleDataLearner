using Test
using SimpleDataLearner
using Zygote
using LinearAlgebra


@testset "gradient test" begin
    a = [1,2]
    f(x,y) = dot(a, [x,y]) + 5
    g = gradient(Params([a])) do
        f(4,5)^2
    end
    @test g.grads[a] == [152.0, 190.0]
end

@testset "Basic tests" begin
    A = Dense(3, 2)
    model = Model([Dense(3,2), RELU, Dense(2,2), Softmax])
    f = x -> SimpleDataLearner.apply(model, x)[1]
    ∇(f, [1,2,3])
    arr = params(model)
    testloss(model) = begin
        v = SimpleDataLearner.apply(model, [1,2,3])
        return dot(v,v)
    end
    grads = gradient(Params(params(model))) do
        testloss(model)
    end
    @test length(arr) == 4
    @test arr[1] == model.components[1].b
    @test arr[2] == model.components[1].W
    @test arr[3] == model.components[3].b
    @test arr[4] == model.components[3].W
end