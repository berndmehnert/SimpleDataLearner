using Test
using SimpleDataLearner

@testset "Basic tests" begin
    A = Dense(3, 2)
    model = Model([Dense(3,2), RELU, Dense(2,2), Softmax])
    f = x -> SimpleDataLearner.apply(model, x)[1]
    ∇(f, [1,2,3])
    @test 1 == 1
end