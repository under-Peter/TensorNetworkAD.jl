using LinalgBackwards, OMEinsum
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
        cp = einsum(((-1,-2),(1,-3,-1),(-2,-4,4),(2,3,-4,-3)),
            (c, t, t, a), (1,2,3,4))
        tp = einsum(((1,-1,5), (3,4,-1,2)), (t,a), (1,2,3,4,5))

        # renormalize
        cpmat = reshape(cp, size(cp,1) * size(cp,2), size(cp,3) * size(cp,4))
        u, s, v = svd(cpmat)
        z = reshape(u[:, 1:χ], χ, d, χ)
        c = einsum(((-1,-2,-3,-4), (-1,-2,1), (-4,-3,2)), (cp, z, conj(z)), (1,2))
        t = einsum(((-1,-2,2,-3,-4), (-1,-2,1), (-4,-3,3)), (tp, conj(z), z), (1,2,3))

        # symmetrize
        c +=  transpose(c)
        t += permutedims(t, (3,2,1))

        # evaluate
        _, vals, = svd(c)
        maxval = maximum(vals)
        c = c ./ maxval
        t = t ./ einsum(((1,2,3),), (t,), ())[1]
        vals = vals ./ norm(vals,1) #normalize(vals,1)

        #compare
        diff = sum(abs, oldsvdvals - vals)
        oldsvdvals = vals
    end
    return (c, t, vals)
end
