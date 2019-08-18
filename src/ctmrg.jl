using LinearAlgebra: normalize, norm, diag
using Random
using IterTools: iterated
using Base.Iterators: take, drop
using Optim, LineSearches

export AbstractLattice, SquareLattice
abstract type AbstractLattice end
struct SquareLattice <: AbstractLattice end

export IPEPS, SquareIPEPS

"""
    IPEPS{LT<:AbstractLattice, T, N}

Infinite projected entangled pair of states.
`LT` is the type of lattice, `T` and `N` are bulk tensor element type and order.
"""
struct IPEPS{LT<:AbstractLattice, T, N}
    a::AbstractArray{T, N}
    # TODO: check input size in constructor
end
IPEPS{LT}(a::AbstractArray{T,N}) where {LT,T,N} = IPEPS{LT,T,N}(a)
nflavor(ipeps::IPEPS) = size(ipeps.a, 1)

const SquareIPEPS{T} = IPEPS{SquareLattice, T, 4}
SquareIPEPS(a::AbstractArray{T, 4}) where T = IPEPS{SquareLattice,T,4}(a)

# NOTE: maybe `C` (corner) and `E` (edge) are better.
struct CTMRGEnv{CT,TT}
    c::CT
    t::TT
    # TODO: check input size in constructor
end

getχ(env::CTMRGEnv) = size(env.c, 1)

"""
    initialize_env(mode, a, χ)

return a χ×χ corner-matrix `c`.
if `randinit == true`, return a random matrix,
otherwise return `a` with two indices summed over
embedded in a χ×χ zeros-matrix.

return a χ×d×χ tensor `t` where `d` is the
dimension of the indices of `a`.
if `randinit == true`, return a random matrix,
otherwise return `a` with two indices summed over
embedded in a χ×d×χ zeros-matrix.
"""
function initialize_env(::Val{:random}, ipeps::SquareIPEPS{T}, χ::Int) where T
    c = randn(T, χ, χ)
    c += adjoint(c)
    t = randn(T, χ, nflavor(ipeps), χ)
    t += permutedims(conj(t), (3,2,1))
    return CTMRGEnv(c, t)
end

# NOTE: what is this initialization strategy?
function initialize_env(::Val{:raw}, ipeps::SquareIPEPS{T}, χ::Int) where T
    c = zeros(T, χ, χ)
    cinit = ein"ijkl -> ij"(ipeps.a)
    foreach(CartesianIndices(cinit)) do i
        i in CartesianIndices(c) && (c[i] = cinit[i])
    end
    t = zeros(T, χ, nflavor(ipeps), χ)
    tinit = ein"ijkl -> ijk"(ipeps.a)
    foreach(CartesianIndices(tinit)) do i
        i in CartesianIndices(t) && (t[i] = tinit[i])
    end
    return CTMRGEnv(c, t)
end

@Zygote.nograd initialize_env

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
                randinit = false)
    a = SquareIPEPS(a)
    env = initialize_env(Val(randinit ? :random : :raw),a,χ)
    ctmrg(a, env; tol=tol, maxit=maxit)
end

function ctmrg(a::SquareIPEPS, env::CTMRGEnv; tol::Real, maxit::Integer)
    d = nflavor(a)
    # initialize
    oldvals = fill(Inf, getχ(env)*d)

    stopfun = StopFunction(oldvals, -1, tol, maxit)
    env, vals = fixedpoint(ctmrgstep, (env, oldvals), (a, getχ(env), d), stopfun)
    return env, vals
end

"""
    ctmrgstep((env,vals), (a, χ, d))

evaluate one step of the ctmrg-algorithm, returning an updated `(c,t,vals)`
which results from growing, renormalizing and symmetrizing `c` and `t` with `a`.
`vals` are the singular values of the grown corner-matrix normalized such that
the leading singular value is 1.
"""
function ctmrgstep((env,vals), (ipeps, χ, d))
    # grow
    a = ipeps.a
    c, t = env.c, env.t
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
    c /= norm(c)
    t /= norm(t)

    return CTMRGEnv(c, t), vals
end
