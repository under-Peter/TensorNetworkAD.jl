using TensorNetworkAD
using Test, Random
using Zygote

@testset "autodiff" begin
    a = randn(10,10)
    @test Zygote.gradient(norm, a)[1] ≈ num_grad(norm, a)

    foo = x -> sum(Float64[x x; x x])
    @test Zygote.gradient(foo, 1)[1] ≈ num_grad(foo, 1)
end
