using BackwardsLinalg, OMEinsum
using LinearAlgebra: normalize, norm, diag
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
    oldvals = fill(Inf, χ*d)

    stopfun = StopFunction(oldvals, -1, tol, maxit)
    c, t, = fixedpoint(ctmrgstep, (cinit, tinit, oldvals), (a, χ, d), stopfun)
    return c, t
end

function ctmrgstep((c,t,vals), (a, χ, d))
    # grow
    # cp = einsum("ad,iba,dcl,jkcb -> ijlk", (c, t, t, a))
    ct = einsum("ad,iba -> dib", (c,t))
    ctt = einsum("dib,dcl -> bcil", (ct,t))
    ctta = einsum("bcil, jkcb -> ijlk", (ctt,a))
    cp = ctta
    tp = einsum("iam,jkla -> ijklm", (t,a))

    # renormalize
    cpmat = reshape(cp, χ*d, χ*d)
    u, s, v = svd(cpmat)
    z = reshape(u[:, 1:χ], χ, d, χ)

    # c = einsum("abcd,abi,cdj -> ij", (cp, conj(z), z))
    cz = einsum("abcd,abi -> cdi", (cp, conj(z)))
    c = einsum("cdi,cdj -> ij", (cz, z))

    # t = einsum("abjcd,abi,dck -> ijk", (tp, conj(z), z))
    tz = einsum("abjcd,abi -> ijcd", (tp, conj(z)))
    t = einsum("ijcd,dck -> ijk", (tz,z))

    vals = s ./ s[1]


    # symmetrize
    c += permutedims(c)
    t += permutedims(t, (3,2,1))

    #gauge fix
    c *= sign(c[1])
    signs = sign.(t[:,2,1])
    # t = einsum("i,ijk,k -> ijk", (signs, t, signs))
    t = t .* reshape(signs,:,1,1) .* reshape(signs,1,1,:)

    # normalize
    c /= norm(c)
    t /= norm(t)

    return c, t, vals
end
