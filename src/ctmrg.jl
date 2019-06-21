using BackwardsLinalg, OMEinsum
using LinearAlgebra: normalize, norm
using Random

initializec(a, χ) = rand(eltype(a), χ, χ)
initializet(a, χ) = rand(eltype(a), χ, size(a,1), χ)
@Zygote.nograd initializec, initializet

function ctmrg(a, χ, tol, maxit::Integer)
    d = size(a,1)
    # initialize
    c = initializec(a, χ)
    t = initializet(a, χ)

    # symmetrize
    c += transpose(c)
    t += permutedims(t, (3,2,1))
    oldsvdvals = Inf .* ones(χ)
    vals = copy(oldsvdvals)
    diff = Inf

    for i in 1:maxit
        # grow
        cp = einsum("ab,ica,bdl,jkdc -> ijkl", (c, t, t, a))
        tp = einsum("iam,klaj -> ijklm", (t,a))

        # renormalize
        cpmat = reshape(cp, size(cp,1) * size(cp,2), size(cp,3) * size(cp,4))
        u, s, v = svd(cpmat)
        z = reshape(u[:, 1:χ], χ, d, χ)
        c = einsum("abcd,abi,dcj -> ij", (cp, z, conj(z)))
        t = einsum("abjcd,abi,dck -> ijk", (tp, conj(z), z))

        # symmetrize
        c +=  transpose(c)
        t += permutedims(t, (3,2,1))

        # evaluate
        _, vals, = svd(c)
        maxval = maximum(vals)
        c = c ./ maxval
        t = t ./ einsum("ijk -> ", (t,))[]
        vals = vals ./ norm(vals,1) #normalize(vals,1)

        #compare
        diff = sum(abs, oldsvdvals - vals)
        oldsvdvals = vals
    end
    return (c, t, vals)
end
