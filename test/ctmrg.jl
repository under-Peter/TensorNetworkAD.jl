using TensorNetworkAD
using TensorNetworkAD: magnetisation, magofβ, initialize_env, getχ, CTMRGEnv, nflavor
using Test, Random
using Zygote

@testset "IPEPS" begin
    @test SquareLattice() isa AbstractLattice
    @test IPEPS{SquareLattice}(randn(2,2,2,2)) isa IPEPS
    sq = SquareIPEPS(randn(2,2,2,2))
    @test sq isa SquareIPEPS
    @test nflavor(sq) == 2

    # env
    # NOTE: initializet and initializec should have been tested.
    env1 = initialize_env(Val(:random), sq, 10)
    @test env1 isa CTMRGEnv
    @test getχ(env1) == 10
    env2 = initialize_env(Val(:raw), sq, 10)
    @test env1 isa CTMRGEnv
    @test getχ(env2) == 10
end

# NOTE: I didn't notice this test should return two values, and made some stupid errors
# This is why I added this naive test.
@testset "ctmrg unit test" begin
    sq = SquareIPEPS(randn(2,2,2,2))
    env = initialize_env(Val(:random), sq, 10)
    env, vals = ctmrg(sq, env; tol=1e-6, maxit=10)
    @test env isa CTMRGEnv
    @test getχ(env) == 10
    @test vals isa Vector
end


@testset "ctmrg" begin
    @test isapprox(magnetisation(Ising(), 0,2), magofβ(0), atol=1e-6)
    @test magnetisation(Ising(), 1,2) ≈ magofβ(1)
    @test isapprox(magnetisation(Ising(), 0.2,10), magofβ(0.2), atol = 1e-4)
    @test isapprox(magnetisation(Ising(), 0.4,10), magofβ(0.4), atol = 1e-3)
    @test magnetisation(Ising(), 0.6,4) ≈ magofβ(0.6)
    @test magnetisation(Ising(), 0.8,2) ≈ magofβ(0.8)

    Random.seed!(1)
    foo = x -> magnetisation(Ising(), x,2)
    @test isapprox(Zygote.gradient(foo,0.5)[1], num_grad(foo,0.5, δ=1e-3), atol = 1e-2)

end
