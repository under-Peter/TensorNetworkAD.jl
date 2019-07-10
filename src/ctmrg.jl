using BackwardsLinalg, OMEinsum
using LinearAlgebra: normalize, norm
using Random
using IterTools: iterated
using Base.Iterators: take, drop

function initializec(a, χ, randinit)
    c = zeros(eltype(a), χ, χ)
    if randinit
        rand!(c)
        c += transpose(c)
    else
        cinit = einsum("ijkl -> ij", (a,))
        foreach(CartesianIndices(cinit)) do i
            i in CartesianIndices(c) && (c[i] = cinit[i])
        end
    end
    return c
end

function initializet(a, χ, randinit)
    t = zeros(eltype(a), χ, size(a,1), χ)
    if randinit
        rand!(t)
        t += permutedims(t, (3,2,1))
    else
        tinit = einsum("ijkl -> ijk", (a,))
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
    oldvals = fill(Inf, χ)

    stopfun = StopFunction(oldvals, 0, tol, maxit)
    c, t, = fixedpoint(ctmrgstep, (cinit, tinit, oldvals), (a, χ, d), stopfun)
    return c, t
end

function ctmrgstep((c,t,vals), (a, χ, d))
    # grow
    cp = einsum("ab,ica,bdl,jkdc -> ijkl", (c, t, t, a))
    tp = einsum("iam,jkla -> ijklm", (t,a))

    # renormalize
    cpmat = reshape(cp, χ*d, χ*d)
    u, s, v = svd(cpmat)
    z = reshape(u[:, 1:χ], χ, d, χ)
    c = einsum("abcd,abi,dcj -> ij", (cp, conj(z), z))
    t = einsum("abjcd,abi,dck -> ijk", (tp, conj(z), z))
    vals = s[1:χ] ./ s[1]

    # symmetrize & normalize
    c += transpose(c)
    t += permutedims(t, (3,2,1))
    c /= norm(c)
    t /= norm(t)

    return c, t, vals
end
