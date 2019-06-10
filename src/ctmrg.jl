using LinalgBackwards, OMEinsum
using LinearAlgebra: normalize
function ctmrg(a, χ, tol, maxit::Integer)
    #=
         |3
      4-[a]-2
         |1

     [c]-2
      |
      1

       |3
      [t]-2
       |1
    =#
    d = size(a,1)
    # initialize
    c = rand(eltype(a), χ, χ)
    t = rand(eltype(a), χ, d, χ)

    # symmetrize
    c += transpose(c)
    t += permutedims(t, (3,2,1))
    oldsvdvals = fill(Inf, χ)
    vals = copy(oldsvdvals)

    for i in 1:maxit
        # grow
        cp = meinsum(((-1,-2),(1,-3,-1),(-2,-4,4),(2,3,-4,-3)),
            (c, t, t, a), (1,2,3,4))
        tp = meinsum(((1,-1,5), (3,4,-1,2)), (t,a), (1,2,3,4,5))

        # renormalize
        cpmat = reshape(cp, prod(size(cp)[1:2]), prod(size(cp)[3:4]))
        u, s, v = svd(cpmat)
        z = reshape(u[:, 1:χ], χ, d, χ)
        c = meinsum(((-1,-2,-3,-4), (-1,-2,1), (-4,-3,2)), (cp, z, conj(z)), (1,2))
        t = meinsum(((-1,-2,2,-3,-4), (-1,-2,1), (-4,-3,3)), (tp, conj(z), z), (1,2,3))

        # symmetrize
        c +=  transpose(c)
        t += permutedims(t, (3,2,1))

        # evaluate
        _, vals, = svd(c)
        maxval = maximum(vals)
        c = c ./ maxval
        t = t ./ meinsum(((1,2,3),), (t,), ())
        vals = normalize(vals,1)

        #compare
        sum(abs, oldsvdvals - vals) < tol && return (c, t, vals)
        oldsvdvals = vals
    end
    return (c, t, vals)
end
