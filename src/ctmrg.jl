using BackwardsLinalg, OMEinsum
using LinearAlgebra: normalize, norm, diag
using Random
using IterTools: iterated
using Base.Iterators: take, drop
using Optim, LineSearches

function initializec(a, χ, randinit)
    # c = zeros(eltype(a), χ, χ)
    c = similar(a, χ, χ)
    if randinit
        rand!(c)
        c += adjoint(c)
    else
        fill!(c, zero(eltype(a)))
        cinit = ein"ijkl -> ij"(a)
        foreach(CartesianIndices(cinit)) do i
            i in CartesianIndices(c) && (c[i] = cinit[i])
        end
    end
    return c
end

function initializet(a, χ, randinit)
    t = similar(a, χ, size(a,1), χ)
    if randinit
        rand!(t)
        t += permutedims(conj(t), (3,2,1))
    else
        tinit = ein"ijkl -> ijk"(a)
        fill!(t, zero(eltype(a)))
        foreach(CartesianIndices(tinit)) do i
            i in CartesianIndices(t) && (t[i] = tinit[i])
        end
    end
    return t
end

@Zygote.nograd initializec, initializet

function ctmrg(a, χ, tol, maxit::Integer, randinit = false)
    d = size(a,1)
    # initialize
    cinit = initializec(a, χ, randinit)
    tinit = initializet(a, χ, randinit)
    oldvals = similar(real(a), χ*d) .* convert(real(eltype(a)),Inf)
    stopfun = StopFunction(oldvals, -1, tol, maxit)

    c, t, = fixedpoint(ctmrgstep, (cinit, tinit, oldvals), (a, χ, d), stopfun)
    return c, t
end

function ctmrgstep((c,t,vals), (a, χ, d))
    # grow
    cp = ein"((ad,iba),dcl),jkcb -> ijlk"(c, t, t, a)
    tp = ein"iam,jkla -> ijklm"(t,a)

    # renormalize
    cpmat = reshape(cp, χ*d, χ*d)
    cpmat += adjoint(cpmat)
    u, s, v = svd(cpmat)
    z = reshape(u[:, 1:χ], χ, d, χ)

    c = ein"(abcd,abi),cdj -> ij"(cp, conj(z), z)
    t = ein"(abjcd,abi),dck -> ijk"(tp, conj(z), z)

    vals = s ./ s[1]


    # symmetrize
    c += adjoint(c)
    t += permutedims(conj(t), (3,2,1))

    # normalize
    c /= mynorm(c)
    t /= mynorm(t)

    return c, t, vals
end

mynorm(x) = sqrt(sum(y -> y * y, x))
