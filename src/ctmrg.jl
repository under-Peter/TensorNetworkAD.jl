using LinearAlgebra: normalize, norm, diag
using Random
using IterTools: iterated
using Base.Iterators: take, drop
using Optim, LineSearches

export AbstractLattice, SquareLattice
abstract type AbstractLattice end
struct SquareLattice <: AbstractLattice end

export CTMRGRuntime, SquareCTMRGRuntime

# NOTE: should be renamed to more explicit names
"""
the size of `a` is `D × D × D × D`
the size of `c` is `χ × χ`
the size of `t` is `χ × D × χ`

Here, `D` is equal to `d^2` as in iPEPS
"""
struct CTMRGRuntime{LT,T,N,AT<:AbstractArray{T,N}}
    a::AT
    c
    t
    function CTMRGRuntime{LT}(a::AT,
        # TODO: check input size in constructor
        c::AbstractArray{T}, t::AbstractArray{T}) where {LT<:AbstractLattice,T,N,AT<:AbstractArray{T,N}}
        new{LT,T,N,AT}(a,c,t)
    end
end
const SquareCTMRGRuntime{T,AT} = CTMRGRuntime{SquareLattice,T,4,AT}
SquareCTMRGRuntime(a::AT,c,t) where {T,AT<:AbstractArray{T, 4}} = CTMRGRuntime{SquareLattice}(a,c,t)

getχ(rt::CTMRGRuntime) = size(rt.c, 1)
getD(rt::CTMRGRuntime) = size(rt.a, 1)

"""
    initialize_env(mode, a, χ)

return a χ×χ corner-matrix `c`.
if `randinit == true`, return a random matrix,
otherwise return `a` with two indices summed over
embedded in a χ×χ zeros-matrix.

return a χ×D×χ tensor `t` where `D` is the
dimension of the indices of `a`.
if `randinit == true`, return a random matrix,
otherwise return `a` with two indices summed over
embedded in a χ×D×χ zeros-matrix.
"""
function SquareCTMRGRuntime(a::AbstractArray{T,4}, env::Val, χ::Int) where T
    return SquareCTMRGRuntime(a, _initializect_square(a, env, χ)...)
end

function _initializect_square(a::AbstractArray{T,4}, env::Val{:random}, χ::Int) where T
    c = randn(T, χ, χ)
    c += adjoint(c)
    t = randn(T, χ, size(a,1), χ)
    t += permutedims(conj(t), (3,2,1))
    c, t
end

function _initializect_square(a::AbstractArray{T,4}, env::Val{:raw}, χ::Int) where T
    c = zeros(T, χ, χ)
    cinit = ein"ijkl -> ij"(a)
    foreach(CartesianIndices(cinit)) do i
        i in CartesianIndices(c) && (c[i] = cinit[i])
    end
    t = zeros(T, χ, size(a,1), χ)
    tinit = ein"ijkl -> ijk"(a)
    foreach(CartesianIndices(tinit)) do i
        i in CartesianIndices(t) && (t[i] = tinit[i])
    end
    c, t
end

@Zygote.nograd _initializect_square
@Zygote.adjoint function CTMRGRuntime{LT}(a::AT,
        c::AbstractArray{T}, t::AbstractArray{T}) where {LT<:AbstractLattice,T,N,AT<:AbstractArray{T,N}}
        return CTMRGRuntime{LT}(a,c,t), dy->(dy.a, dy.c, dy.t)
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
function ctmrg(rt::CTMRGRuntime; tol::Real, maxit::Integer)
    # initialize
    oldvals = fill(Inf, getχ(rt)*getD(rt))

    stopfun = StopFunction(oldvals, -1, tol, maxit)
    rt, vals = fixedpoint(res->ctmrgstep(res...), (rt, oldvals), stopfun)
    return rt, vals
end

"""
    ctmrgstep(rt,vals)

evaluate one step of the ctmrg-algorithm, returning an updated `(c,t,vals)`
which results from growing, renormalizing and symmetrizing `c` and `t` with `a`.
`vals` are the singular values of the grown corner-matrix normalized such that
the leading singular value is 1.
"""
function ctmrgstep(rt::SquareCTMRGRuntime, vals)
    # grow
    a, c, t = rt.a, rt.c, rt.t
    D, χ = getD(rt), getχ(rt)
    cp = ein"ad,iba,dcl,jkcb -> ijlk"(c, t, t, a)
    tp = ein"iam,jkla -> ijklm"(t,a)

    # renormalize
    cpmat = reshape(cp, χ*D, χ*D)
    cpmat += adjoint(cpmat)
    u, s, v = svd(cpmat)
    z = reshape(u[:, 1:χ], χ, D, χ)

    c = ein"abcd,abi,cdj -> ij"(cp, conj(z), z)
    t = ein"abjcd,abi,dck -> ijk"(tp, conj(z), z)

    vals = s ./ s[1]

    # indexperm_symmetrize
    c += c'
    t += ein"ijk -> kji"(conj(t))

    # normalize
    c /= norm(c)
    t /= norm(t)

    return SquareCTMRGRuntime(a, c, t), vals
end
