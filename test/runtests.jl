using Test
using SimpleDataLearner

@testset "Basic tests" begin
    A = Dense(3, 2)
    model = Model([Dense(3,2), RELU, Dense(2,2), Softmax])
    f = x -> SimpleDataLearner.apply(model, x)[1]
    ∇(f, [1,2,3])
    arr = params(model)
    @test length(arr) == 4
    @test arr[1] == model.components[1].b
    @test arr[2] == model.components[1].W
    @test arr[3] == model.components[3].b
    @test arr[4] == model.components[3].W
end