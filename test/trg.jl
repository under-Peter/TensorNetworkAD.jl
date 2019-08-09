using TensorNetworkAD
using Test
using Zygote
using TensorNetworkAD: isingtensor, tensorfromclassical

@testset "trg" begin
    @testset "exampletensor" begin
        β = rand()
        @isdefined(pmobj) && next!(pmobj)
        @test isingtensor(β) ≈ tensorfromclassical([β -β; -β β])
    end
    @testset "real" begin
        χ, niter = 5, 5
        foo = β -> trg(isingtensor(β), χ, niter)
        # the pytorch result with tensorgrad
        # https://github.com/wangleiphy/tensorgrad
        # clone this repo and type
        # $ python 1_ising_TRG/ising.py -chi 5 -Niter 5
        @isdefined(pmobj) && next!(pmobj)
        @test foo(0.4) ≈ 0.8919788686747141
        @isdefined(pmobj) && next!(pmobj)
        @test num_grad(foo, 0.4, δ=1e-6) ≈ Zygote.gradient(foo, 0.4)[1]
    end

    @testset "complex" begin
        β, χ, niter = 0.4, 12, 3
        @isdefined(pmobj) && next!(pmobj)
        @test trg(isingtensor(β), χ, niter) ≈
            real(trg(isingtensor(β) .+ 0im, χ, niter))
        @isdefined(pmobj) && next!(pmobj)
        @test Zygote.gradient(β -> trg(isingtensor(β), χ, niter), 0.4)[1] ≈
            real(Zygote.gradient(β -> real(trg(isingtensor(β) .+ 0im, χ, niter)), 0.4)[1])
    end
end