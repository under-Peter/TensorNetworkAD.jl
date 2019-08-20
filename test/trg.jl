using TensorNetworkAD
using Test
using Zygote, OMEinsum
using TensorNetworkAD: model_tensor, tensorfromclassical, trg_svd

@testset "unit" begin
    t = randn(10,10,10,10)
    u, v = trg_svd(t, 100, 0)
    @test t ≈ ein"ija,akl -> ijkl"(u,v)
end
@testset "real" begin
    χ, niter = 5, 5
    foo = β -> trg(model_tensor(Ising(),β), χ, niter)
    # the pytorch result with tensorgrad
    # https://github.com/wangleiphy/tensorgrad
    # clone this repo and type
    # $ python 1_ising_TRG/ising.py -chi 5 -Niter 5
    @test foo(0.4) ≈ 0.8919788686747141
    @test num_grad(foo, 0.4, δ=1e-6) ≈ Zygote.gradient(foo, 0.4)[1]
end

@testset "complex" begin
    β, χ, niter = 0.4, 12, 3
    @test trg(model_tensor(Ising(),β), χ, niter) ≈
        real(trg(model_tensor(Ising(),β) .+ 0im, χ, niter))
    @test Zygote.gradient(β -> trg(model_tensor(Ising(),β), χ, niter), 0.4)[1] ≈
        real(Zygote.gradient(β -> real(trg(model_tensor(Ising(),β) .+ 0im, χ, niter)), 0.4)[1])
end
