using TensorNetworkAD
using TensorNetworkAD: magnetisation, magofβ, getχ, CTMRGRuntime, getd, gets
using Test, Random
using Zygote

@testset "IPEPS" begin
    @test SquareLattice() isa AbstractLattice
    @test IPEPS{SquareLattice}(randn(2,3,3,3,3)) isa IPEPS
    sq = SquareIPEPS(randn(3,3,3,3,2))
    @test sq isa SquareIPEPS
    @test getd(sq) == 3
    @test gets(sq) == 2
    @test_throws DimensionMismatch SquareIPEPS(randn(3,3,4,3,2))
end

@testset "runtime" begin
    # NOTE: previous initializet and initializec should have been tested.
    a = randn(ComplexF64, 2,2,2,2)
    env1 = SquareCTMRGRuntime(a, Val(:random), 10)
    @test env1 isa CTMRGRuntime
    @test getχ(env1) == 10
    env2 = SquareCTMRGRuntime(a, Val(:raw), 10)
    @test env1 isa CTMRGRuntime
    @test getχ(env2) == 10
end

@testset "ctmrg unit test" begin
    rt = SquareCTMRGRuntime(randn(2,2,2,2), Val(:random), 10)
    rt = ctmrg(rt; tol=1e-6, maxit=10)
    @test rt isa CTMRGRuntime
    @test getχ(rt) == 10
end


@testset "ctmrg" begin
    Random.seed!(5)
    @test isapprox(magnetisation(Ising(), 0,2), magofβ(Ising(),0), atol=1e-6)
    @test magnetisation(Ising(), 1,2) ≈ magofβ(Ising(),1)
    @test isapprox(magnetisation(Ising(), 0.2,10), magofβ(Ising(),0.2), atol = 1e-4)
    @test isapprox(magnetisation(Ising(), 0.4,10), magofβ(Ising(),0.4), atol = 1e-3)
    @test magnetisation(Ising(), 0.6,4) ≈ magofβ(Ising(),0.6)
    @test magnetisation(Ising(), 0.8,2) ≈ magofβ(Ising(),0.8)

    Random.seed!(9)
    foo = x -> magnetisation(Ising(), x,2)
    @test isapprox(Zygote.gradient(foo,0.5)[1], num_grad(foo,0.5, δ=1e-3), atol = 1e-2)
end
