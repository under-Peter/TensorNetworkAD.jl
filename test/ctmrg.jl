using TensorNetworkAD
using TensorNetworkAD: magnetisationofβ, magofβ
using Test, Random
using Zygote


@testset "ctmrg" begin
    @isdefined(pmobj) && next!(pmobj)
    @test isapprox(magnetisationofβ(0,2), magofβ(0), atol=1e-6)
    @isdefined(pmobj) && next!(pmobj)
    @test magnetisationofβ(1,2) ≈ magofβ(1)
    @isdefined(pmobj) && next!(pmobj)
    @test isapprox(magnetisationofβ(0.2,10), magofβ(0.2), atol = 1e-4)
    @isdefined(pmobj) && next!(pmobj)
    @test isapprox(magnetisationofβ(0.4,10), magofβ(0.4), atol = 1e-3)
    @isdefined(pmobj) && next!(pmobj)
    @test magnetisationofβ(0.6,4) ≈ magofβ(0.6)
    @isdefined(pmobj) && next!(pmobj)
    @test magnetisationofβ(0.8,2) ≈ magofβ(0.8)

    Random.seed!(1)
    foo = x -> magnetisationofβ(x,2)
    @isdefined(pmobj) && next!(pmobj)
    @test isapprox(Zygote.gradient(foo,0.5)[1], num_grad(foo,0.5, δ=1e-3), atol = 1e-2)

end
