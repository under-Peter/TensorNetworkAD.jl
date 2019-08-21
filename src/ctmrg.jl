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
    CTMRGRuntime{LT}

a struct to hold the tensors during the `ctmrg` algorithm, containing
- `D × D × D × D` `bulk` tensor
- `χ × χ` `corner` tensor
- `χ × D × χ` `edge` tensor
and `LT` is a AbstractLattice to define the lattice type.
"""
struct CTMRGRuntime{LT,T,N,AT<:AbstractArray{T,N},CT,ET}
    bulk::AT
    corner::CT
    edge::ET
    function CTMRGRuntime{LT}(bulk::AT,
        # TODO: check input size in constructor
        corner::AbstractArray{T}, edge::AbstractArray{T}) where {LT<:AbstractLattice,T,N,AT<:AbstractArray{T,N}}
        new{LT,T,N,AT,typeof(corner), typeof(edge)}(bulk,corner,edge)
    end
end
const SquareCTMRGRuntime{T,AT} = CTMRGRuntime{SquareLattice,T,4,AT}
SquareCTMRGRuntime(bulk::AT,corner,edge) where {T,AT<:AbstractArray{T, 4}} = CTMRGRuntime{SquareLattice}(bulk,corner,edge)

getχ(rt::CTMRGRuntime) = size(rt.corner, 1)
getD(rt::CTMRGRuntime) = size(rt.bulk, 1)

@doc raw"
    SquareCTMRGRuntime(bulk::AbstractArray{T,4}, env::Val, χ::Int)

create a `SquareCTMRGRuntime` with bulk-tensor `bulk`. The corner and edge
tensors are initialized according to `env`. If `env = Val(:random)`,
the corner is initialized as a random χ×χ tensor and the edge is initialized
as a random χ×D×χ tensor where `D = size(bulk,1)`.
If `env = Val(:raw)`, corner- and edge-tensor are initialized by summing
over one or two indices of `bulk` respectively and embedding the result
in zeros-tensors of the appropriate size, truncating if necessary.

# example

```jldoctest; setup = :(using TensorNetworkAD)
julia> rt = SquareCTMRGRuntime(randn(2,2,2,2), Val(:raw), 4);

julia> rt.corner[1:2,1:2] ≈ dropdims(sum(rt.bulk, dims = (3,4)), dims = (3,4))
true

julia> rt.edge[1:2,1:2,1:2] ≈ dropdims(sum(rt.bulk, dims = 4), dims = 4)
true
```
"
function SquareCTMRGRuntime(bulk::AbstractArray{T,4}, env::Val, χ::Int) where T
    return SquareCTMRGRuntime(bulk, _initializect_square(bulk, env, χ)...)
end

function _initializect_square(bulk::AbstractArray{T,4}, env::Val{:random}, χ::Int) where T
    corner = randn(T, χ, χ)
    edge = randn(T, χ, size(bulk,1), χ)
    corner += adjoint(corner)
    edge += permutedims(conj(edge), (3,2,1))
    corner, edge
end

function _initializect_square(bulk::AbstractArray{T,4}, env::Val{:raw}, χ::Int) where T
    corner = zeros(T, χ, χ)
    edge = zeros(T, χ, size(bulk,1), χ)
    cinit = ein"ijkl -> ij"(bulk)
    tinit = ein"ijkl -> ijk"(bulk)
    foreach(CartesianIndices(cinit)) do i
        i in CartesianIndices(corner) && (corner[i] = cinit[i])
    end
    foreach(CartesianIndices(tinit)) do i
        i in CartesianIndices(edge) && (edge[i] = tinit[i])
    end
    corner, edge
end


"""
    ctmrg(rt::CTMRGRuntime; tol, maxit)

return a `CTMRGRuntime` with an environment consisting of
corner and edge tensor that have either been iterated for `maxit` iterations
or converged according to `tol`.
Convergence is tested by looking at the sum of the absolut differences in the
corner singular values. If it is less than `tol`, convergence is reached.

# example
```
julia> a = model_tensor(Ising(),β);

julia> rt = SquareCTMRGRuntime(a, Val(:random), χ);

julia> env = ctmrg(rt; tol=1e-6, maxit=100);
```

for the environment of an isingmodel at inverse temperature β
on an infinite square lattice.
"""
function ctmrg(rt::CTMRGRuntime; tol::Real, maxit::Integer)
    # initialize
    oldvals = fill(Inf, getχ(rt)*getD(rt))

    stopfun = StopFunction(oldvals, -1, tol, maxit)
    rt, vals = fixedpoint(res->ctmrgstep(res...), (rt, oldvals), stopfun)
    return rt
end

"""
    ctmrgstep(rt,vals)

evaluate one step of the ctmrg-algorithm, returning a tuple of an updated `CTMRGRuntime`
with updated `corner` and `edge` tensor and a vector of singular values to test
convergence with.
"""
function ctmrgstep(rt::SquareCTMRGRuntime, vals)
    # grow
    bulk, corner, edge = rt.bulk, rt.corner, rt.edge
    D, χ = getD(rt), getχ(rt)
    cp = ein"ad,iba,dcl,jkcb -> ijlk"(corner, edge, edge, bulk)
    tp = ein"iam,jkla -> ijklm"(edge,bulk)

    # renormalize
    cpmat = reshape(cp, χ*D, χ*D)
    cpmat += adjoint(cpmat)
    u, s, v = svd(cpmat)
    z = reshape(u[:, 1:χ], χ, D, χ)

    corner = ein"abcd,abi,cdj -> ij"(cp, conj(z), z)
    edge = ein"abjcd,abi,dck -> ijk"(tp, conj(z), z)

    vals = s ./ s[1]

    # indexperm_symmetrize
    corner += corner'
    edge += ein"ijk -> kji"(conj(edge))

    # normalize
    corner /= norm(corner)
    edge /= norm(edge)

    return SquareCTMRGRuntime(bulk, corner, edge), vals
end
