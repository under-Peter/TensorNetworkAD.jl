using LinearAlgebra: normalize, norm, diag
using Random
using IterTools: iterated
using Base.Iterators: take, drop
using Optim, LineSearches

"""
    initializec(a, χ, randinit)
return a χ×χ corner-matrix `c`.
if `randinit == true`, return a random matrix,
otherwise return `a` with two indices summed over
embedded in a χ×χ zeros-matrix.
"""
function initializec(a, χ, randinit)
    c = zeros(eltype(a), χ, χ)
    if randinit
        rand!(c)
        c += adjoint(c)
    else
        cinit = ein"ijkl -> ij"(a)
        foreach(CartesianIndices(cinit)) do i
            i in CartesianIndices(c) && (c[i] = cinit[i])
        end
    end
    return c
end

"""
    initializet(a, χ, randinit)

return a χ×d×χ tensor `t` where `d` is the
dimension of the indices of `a`.
if `randinit == true`, return a random matrix,
otherwise return `a` with two indices summed over
embedded in a χ×d×χ zeros-matrix.
"""
function initializet(a, χ, randinit)
    t = zeros(eltype(a), χ, size(a,1), χ)
    if randinit
        rand!(t)
        t += permutedims(conj(t), (3,2,1))
    else
        tinit = ein"ijkl -> ijk"(a)
        foreach(CartesianIndices(tinit)) do i
            i in CartesianIndices(t) && (t[i] = tinit[i])
        end
    end
    return t
end


"""
    ctmrg(a, χ, tol, maxit::Integer, cinit = nothing, tinit = nothing, randinit = false)
returns a tuple `(c,t)` where `c` is the corner-transfer matrix of `a`
and `t` is the half-infinite column/row tensor of `a`.
`a` is assumed to satisfy symmetries w.r.t all possible permutations of
its indices.
Initial values for `c` and `t` can be provided via keyword-arguments.
If no values are provided, `c` and `t` are initialized with random
values (if `randinit = true`) or by reducing over indices of `a` (otherwise).

The ctmrg-algorithm is run for up to `maxit` iterations with
bond-dimension `χ` for the environment. If the sum of absolute
differences between `c`s singular values between two steps is
below `tol` the algorithm is assumed to be converged.
"""
function ctmrg(a::AbstractArray{<:Any,4}, χ::Integer, tol::Real, maxit::Integer;
                cinit = nothing, tinit = nothing, randinit = false)
    d = size(a,1)
    # initialize
    cinit === nothing && (cinit = initializec(a, χ, randinit))
    tinit === nothing && (tinit = initializet(a, χ, randinit))
    oldvals = fill(Inf, χ*d)

    stopfun = StopFunction(oldvals, -1, tol, maxit)
    c, t, = fixedpoint(ctmrgstep, (cinit, tinit, oldvals), (a, χ, d), stopfun)
    return c, t
end

"""
    ctmrgstep((c,t,vals), (a, χ, d))

evaluate one step of the ctmrg-algorithm, returning an updated `(c,t,vals)`
which results from growing, renormalizing and symmetrizing `c` and `t` with `a`.
`vals` are the singular values of the grown corner-matrix normalized such that
the leading singular value is 1.
"""
function ctmrgstep((c,t,vals), (a, χ, d))
    # grow
    cp = ein"ad,iba,dcl,jkcb -> ijlk"(c, t, t, a)
    tp = ein"iam,jkla -> ijklm"(t,a)

    # renormalize
    cpmat = reshape(cp, χ*d, χ*d)
    cpmat += adjoint(cpmat)
    u, s, v = svd(cpmat)
    z = reshape(u[:, 1:χ], χ, d, χ)

    c = ein"abcd,abi,cdj -> ij"(cp, conj(z), z)
    t = ein"abjcd,abi,dck -> ijk"(tp, conj(z), z)

    vals = s ./ s[1]


    # indexperm_symmetrize
    c += c'
    t += ein"ijk -> kji"(conj(t))

    # normalize
    c /= mynorm(c)
    t /= mynorm(t)

    return c, t, vals
end

"""
    mynorm(x)
return the 2-norm of `x` - workaround errors in Zygote.
"""
mynorm(x) = sqrt(sum(y -> y * y, x))
