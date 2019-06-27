using Test
using TensorNetworkAD
using TensorNetworkAD: diaglocalhamiltonian, energy, expectationvalue, optimiseipeps,
                       tfisinghamiltonian
using OMEinsum
using LinearAlgebra: svd

@testset "variationalipeps" begin
    @testset "non-interacting" begin
        h = diaglocalhamiltonian([1,-1])
        as = (rand(3,3,3,3,2) for _ in 1:100)
        @test all(a -> -1 < energy(h,a,5,0,10)/2 < 1, as)

        h = diaglocalhamiltonian([1,-1])
        a = zeros(2,2,2,2,2)
        a[1,1,1,1,2] = randn()
        @test energy(h,a,10,0,300)/2 ≈ -1

        a = zeros(2,2,2,2,2)
        a[1,1,1,1,1] = randn()
        @test energy(h,a,10,0,300)/2 ≈ 1

        a = zeros(2,2,2,2,2)
        a[1,1,1,1,2] = a[1,1,1,1,1] = randn()
        @test abs(energy(h,a,10,0,300)) < 1e-9


        hdiag = [0.3,0.1,-0.43]
        h = diaglocalhamiltonian(hdiag)
        a = randn(2,2,2,2,3)
        res = optimiseipeps(a, h, 4, 0, 100)
        e = energy(h,res.minimizer, 10,0,300)/2
        @test isapprox(e, minimum(hdiag), atol=1e-3)
    end

    @testset "ising" begin
        h = zeros(2,2,2,2)
        h[1,1,2,2] = h[2,2,1,1] = 1
        h[2,2,2,2] = h[1,1,1,1] = -1
        a = randn(2,2,2,2,2)
        res = optimiseipeps(a, h, 4, 0, 100)
        e = energy(h,res.minimizer, 10,0,300)
        @test isapprox(e,-1, atol=1e-3)

        h = zeros(2,2,2,2)
        h[1,1,2,2] = h[2,2,1,1] = 1
        h[2,2,2,2] = h[1,1,1,1] = -1
        randu, s,  = svd(randn(2,2))
        h = einsum("abcd,ai,bj,ck,dl -> ijkl", (h,randu,randu',randu,randu'))
        a = randn(2,2,2,2,2)
        res = optimiseipeps(a, h, 5, 0, 100)
        e = energy(h,res.minimizer, 10,0,300)
        @test isapprox(e,-1, atol=1e-3)

        h = tfisinghamiltonian(1.0)
        a = randn(2,2,2,2,2)
        res = optimiseipeps(a, h, 5, 0, 100)
        e = energy(h,res.minimizer, 5,0,100)
        @test isapprox(e, -2.12566, atol = 1e-3)

        h = tfisinghamiltonian(0.5)
        a = randn(2,2,2,2,2)
        res = optimiseipeps(a, h, 5, 0, 100)
        e = energy(h,res.minimizer, 5,0,100)
        @test isapprox(e, -2.0312, atol = 1e-2)

        h = tfisinghamiltonian(2.0)
        a = randn(2,2,2,2,2)
        res = optimiseipeps(a, h, 5, 0, 100)
        e = energy(h,res.minimizer, 5,0,100)
        @test isapprox(e, -2.5113, atol = 1e-3)
    end
end