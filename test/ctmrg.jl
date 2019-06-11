using TensorNetworkAD
using Test, Random
using Zygote
using OMEinsum

@Zygote.adjoint function Base.typed_hvcat(::Type{T}, rows::Tuple{Vararg{Int}}, xs::S...) where {T,S}
  Base.typed_hvcat(T,rows, xs...), ȳ -> (nothing, nothing, permutedims(ȳ)...)
end


function isingtensor(β)
    a = reshape(Float64[1 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 1], 2,2,2,2)
    cβ, sβ = sqrt(cosh(β)), sqrt(sinh(β))
    q = 1/sqrt(2) * [cβ+sβ cβ-sβ; cβ-sβ cβ+sβ]
    einsum(((-1,-2,-3,-4), (-1,1), (-2,2), (-3,3), (-4,4)), (a,q,q,q,q), (1,2,3,4))
end

function isingmagtensor(β)
    a = reshape([1 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 -1] , 2,2,2,2)
    cβ, sβ = sqrt(cosh(β)), sqrt(sinh(β))
    q = 1/sqrt(2) * [cβ+sβ cβ-sβ; cβ-sβ cβ+sβ]
    einsum(((-1,-2,-3,-4), (-1,1), (-2,2), (-3,3), (-4,4)), (a,q,q,q,q), (1,2,3,4))
end

function magnetisationofβ(β, χ)
    a = isingtensor(β)
    m = isingmagtensor(β)
    c, t, = TensorNetworkAD.ctmrg(a, χ, 1e-6, 100)
    ctc = einsum(((1,-1),(-1,2,-2),(-2,3)), (c,t,c), (1,2,3))
    env = einsum(((-1,4,-3),(-3,3,-4),(-2,2,-4),(-2,1,-1)),(ctc,t,ctc,t),(1,2,3,4))
    mag = einsum(((1,2,3,4),(1,2,3,4)), (env,m),())[]
    norm = einsum(((1,2,3,4),(1,2,3,4)), (env,a),())[]

    return abs(mag/norm)
end

function eoeofβ(β,χ)
    a = isingtensor(β)
    m = isingmagtensor(β)
    c, t, = TensorNetworkAD.ctmrg(a, χ, 1e-6, 100)
    d = diag(c)
    d = d ./ norm(d,1)
    return -sum( d .* log2.(d))
end

function magofβ(β)
    βc = log(1+sqrt(2))/2
    if β > βc
        (1-sinh(2*β)^-4)^(1/8)
    else
        0
    end
end

@testset "ctmrg" begin
    Random.seed!(1)
    @test magnetisationofβ(0,2) ≈ magofβ(0)
    @test magnetisationofβ(1,2) ≈ magofβ(1)
    @test isapprox(magnetisationofβ(0.2,10), magofβ(0.2), atol = 1e-4)
    @test isapprox(magnetisationofβ(0.4,10), magofβ(0.4), atol = 1e-3)
    @test magnetisationofβ(0.6,4) ≈ magofβ(0.6)
    @test magnetisationofβ(0.8,2) ≈ magofβ(0.8)

    foo = x -> magnetisationofβ(x,2)
    @test Zygote.gradient(foo,0.5)[1] ≈ num_grad(foo,0.5)

    # Random.seed!(2)
    # χ = 5
    # niter = 5
    # $ python 2_variational_iPEPS/variational.py -model TFIM -D 3 -chi 10 -Niter 10 -Nepochs 10
    # @test_broken isapprox(ctmrg(:TFIM; nepochs=10, χ=10, D=3, niter=10).E, -2.1256619, atol=1e-5)
end
