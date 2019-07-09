using BackwardsLinalg, OMEinsum
using LinearAlgebra: normalize, norm
using Random

function initializec(a, χ, randinit)
    if randinit
        c = rand(eltype(a), χ, χ)
        c += transpose(c)
        return c
    end
    c = zeros(eltype(a), χ, χ)
    cinit = einsum("ijkl -> ij", (a,))
    foreach(CartesianIndices(cinit)) do i
        i in CartesianIndices(c) && (c[i] = cinit[i])
    end
    return c
end

function initializet(a, χ, randinit)
    if randinit
        t = rand(eltype(a), χ, size(a,1), χ)
        t += permutedims(t, (3,2,1))
        return t
    end
    t = zeros(eltype(a), χ, size(a,1), χ)
    tinit = einsum("ijkl -> ijk", (a,))
    foreach(CartesianIndices(tinit)) do i
        i in CartesianIndices(t) && (t[i] = tinit[i])
    end
    return t
end

@Zygote.nograd initializec, initializet

function ctmrg(a, χ, tol, maxit::Integer, randinit = false)
    d = size(a,1)
    # initialize
    c = initializec(a, χ, randinit)
    t = initializet(a, χ, randinit)

    # symmetrize
    oldsvdvals = Inf .* ones(χ)
    vals = copy(oldsvdvals)
    diff = Inf

    for i in 1:maxit
        # grow
        cp = einsum("ab,ica,bdl,jkdc -> ijkl", (c, t, t, a))
        tp = einsum("iam,jkla -> ijklm", (t,a))

        # renormalize
        cpmat = reshape(cp, χ*d, χ*d)
        u, s, v = svd(cpmat)
        z = reshape(u[:, 1:χ], χ, d, χ)
        c = einsum("abcd,abi,dcj -> ij", (cp, conj(z), z))
        t = einsum("abjcd,abi,dck -> ijk", (tp, conj(z), z))

        # symmetrize
        c += transpose(c)
        t += permutedims(t, (3,2,1))

        # evaluate
        vals = s[1:χ]
        c /= norm(c)
        t /= norm(t)
        vals /= maximum(vals)

        #compare
        diff = sum(abs, oldsvdvals - vals)
        oldsvdvals = vals
    end
    return (c, t, vals)
end
