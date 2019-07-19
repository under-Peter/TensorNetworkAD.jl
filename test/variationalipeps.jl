using Test
using TensorNetworkAD
using TensorNetworkAD: diaglocalhamiltonian, energy, expectationvalue, optimiseipeps,
                       tfisinghamiltonian, heisenberghamiltonian,
                       rotsymmetrize, isrotsym, num_grad
using OMEinsum, Zygote, Random
using LinearAlgebra: svd, norm
using Optim

@testset "variationalipeps" begin
    @testset "non-interacting" begin
        h = diaglocalhamiltonian([1,-1])
        as = (rand(3,3,3,3,2) for _ in 1:100)
        next!(pmobj)
        @test all(a -> -1 < energy(h,a,5,0,10)/2 < 1, as)

        h = diaglocalhamiltonian([1,-1])
        a = zeros(2,2,2,2,2)
        a[1,1,1,1,2] = randn()
        next!(pmobj)
        @test energy(h,a,10,0,300)/2 ≈ -1

        a = zeros(2,2,2,2,2)
        a[1,1,1,1,1] = randn()
        next!(pmobj)
        @test energy(h,a,10,0,300)/2 ≈ 1

        a = zeros(2,2,2,2,2)
        a[1,1,1,1,2] = a[1,1,1,1,1] = randn()
        next!(pmobj)
        @test abs(energy(h,a,10,0,300)) < 1e-9


        hdiag = [0.3,0.1,-0.43]
        h = diaglocalhamiltonian(hdiag)
        a = randn(2,2,2,2,3)
        res = optimiseipeps(a, h, 4, 0, 100,
            optimargs = (Optim.Options(f_tol=1e-6, show_trace=false),))
        e = minimum(res)/2
        next!(pmobj)
        @test isapprox(e, minimum(hdiag), atol=1e-3)
    end

    @testset "ising" begin
        h = zeros(2,2,2,2)
        h[1,1,2,2] = h[2,2,1,1] = 1
        h[2,2,2,2] = h[1,1,1,1] = -1
        a = randn(2,2,2,2,2)
        res = optimiseipeps(a, h, 4, 0, 100,
            optimargs = (Optim.Options(f_tol=1e-6, show_trace=false),))
        e = minimum(res)
        next!(pmobj)
        @test isapprox(e,-1, atol=1e-3)

        h = zeros(2,2,2,2)
        h[1,1,2,2] = h[2,2,1,1] = 1
        h[2,2,2,2] = h[1,1,1,1] = -1
        randu, s,  = svd(randn(2,2))
        h = einsum("abcd,ai,bj,ck,dl -> ijkl", (h,randu,randu',randu,randu'))
        a = randn(2,2,2,2,2)
        res = optimiseipeps(a, h, 6, 0, 200,
            optimargs = (Optim.Options(f_tol=1e-6, show_trace=false),))
        e = minimum(res)
        next!(pmobj)
        @test isapprox(e,-1, atol=1e-3)

        # comparison with results from https://github.com/wangleiphy/tensorgrad
        h = tfisinghamiltonian(1.0)
        a = randn(2,2,2,2,2)
        res = optimiseipeps(a, h, 5, 0, 100,
            optimargs = (Optim.Options(f_tol=1e-6, show_trace=false),))
        e = minimum(res)
        next!(pmobj)
        @test isapprox(e, -2.12566, atol = 1e-3)

        h = tfisinghamiltonian(0.5)
        a = randn(2,2,2,2,2)
        res = optimiseipeps(a, h, 5, 0, 100,
            optimargs = (Optim.Options(f_tol=1e-6, show_trace=false),))
        e = minimum(res)
        next!(pmobj)
        @test isapprox(e, -2.0312, atol = 1e-2)

        Random.seed!(0)
        h = tfisinghamiltonian(2.0)
        a = randn(2,2,2,2,2)
        res = optimiseipeps(a, h, 6, 1e-9, 100,
            optimargs = (Optim.Options(f_tol=1e-8, show_trace=false),))
        e = minimum(res)
        next!(pmobj)
        @test isapprox(e, -2.5113, atol = 1e-3)
    end

    @testset "heisenberg" begin
        # comparison with results from https://github.com/wangleiphy/tensorgrad
        h = heisenberghamiltonian(Jz = 1.)
        a = randn(2,2,2,2,2)
        res = optimiseipeps(a, h, 5, 0, 100,
            optimargs = (Optim.Options(f_tol=1e-6, show_trace=false),))
        e = minimum(res)
        next!(pmobj)
        @test isapprox(e, -0.66023, atol = 1e-3)

        # Random.seed!(0)
        h = heisenberghamiltonian(Jx = 2., Jy = 2.)
        a = randn(2,2,2,2,2)
        res = optimiseipeps(a, h, 6, 0, 100, #optimmethod = Optim.LBFGS(),
            optimargs = (Optim.Options(f_tol = 1e-6, show_trace = false),))
        e = minimum(res)
        next!(pmobj)
        @test isapprox(e, -1.190, atol = 1e-2)

        h = heisenberghamiltonian(Jx = 0.5, Jy = 0.5, Jz = 2.0)
        a = randn(2,2,2,2,2)
        res = optimiseipeps(a, h, 5, 0, 100,
            optimargs = (Optim.Options(f_tol = 1e-6, show_trace = false),))
        e = minimum(res)
        next!(pmobj)
        @test isapprox(e, -1.0208, atol = 1e-3)
    end

    @testset "gradient" begin
        Random.seed!(0)
        h = heisenberghamiltonian()
        a = rotsymmetrize(randn(2,2,2,2,2))
        gradzygote = first(Zygote.gradient(a) do x
            energy(h,x,4,0,100)
        end)
        gradnum = num_grad(a, δ=1e-3) do x
            energy(h,x,4,0,100)
        end

        next!(pmobj)
        @test isapprox(gradzygote, gradnum, atol=1e-3)
    end
end
