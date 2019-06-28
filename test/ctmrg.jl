using TensorNetworkAD
using TensorNetworkAD: magnetisationofβ, magofβ
using Test, Random
using Zygote


@testset "ctmrg" begin
    Random.seed!(1)
    next!(p)
    @test isapprox(magnetisationofβ(0,2), magofβ(0), atol=1e-6)
    next!(p)
    @test magnetisationofβ(1,2) ≈ magofβ(1)
    next!(p)
    @test isapprox(magnetisationofβ(0.2,10), magofβ(0.2), atol = 1e-4)
    next!(p)
    @test isapprox(magnetisationofβ(0.4,10), magofβ(0.4), atol = 1e-3)
    next!(p)
    @test magnetisationofβ(0.6,4) ≈ magofβ(0.6)
    next!(p)
    @test magnetisationofβ(0.8,2) ≈ magofβ(0.8)

    foo = x -> magnetisationofβ(x,2)
    next!(p)
    @test Zygote.gradient(foo,0.5)[1] ≈ num_grad(foo,0.5)

    # Random.seed!(2)
    # χ = 5
    # niter = 5
    # $ python 2_variational_iPEPS/variational.py -model TFIM -D 3 -chi 10 -Niter 10 -Nepochs 10
    # @test_broken isapprox(ctmrg(:TFIM; nepochs=10, χ=10, D=3, niter=10).E, -2.1256619, atol=1e-5)
end
