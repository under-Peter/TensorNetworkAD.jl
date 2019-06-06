using TensorNetworkAD
using Test
using Zygote

@testset "trg" begin
    χ = 5
    niter = 5
    foo = x -> trg(x, χ, niter)
    @test foo(0.4)/2^niter ≈ 0.8919788686747141  # the pytorch result
    @test num_grad(foo, 0.4, δ=1e-6) ≈ Zygote.gradient(foo, 0.4)[1]
end

@testset "trg" begin
    χ = 5
    niter = 5
    foo = x -> trg(x, χ, niter)
    @test foo(0.4)/2^niter ≈ 0.8919788686747141  # the pytorch result
    @test num_grad(foo, 0.4, δ=1e-6) ≈ Zygote.gradient(foo, 0.4)[1]
end
