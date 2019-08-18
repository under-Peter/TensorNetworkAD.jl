export IPEPS, SquareIPEPS
"""
    IPEPS{LT<:AbstractLattice, T, N}

Infinite projected entangled pair of states.
`LT` is the type of lattice, `T` and `N` are bulk tensor element type and order.
"""
struct IPEPS{LT<:AbstractLattice, T, N, AT<:AbstractArray{T, N}}
    t::AT
    # TODO: check input size in constructor
end
IPEPS{LT}(t::AT) where {LT,T,N,AT<:AbstractArray{T,N}} = IPEPS{LT,T,N,AT}(t)

############### IPEPS on square lattice ###################
# size of t is `d × d × d × d × s`
const SquareIPEPS{T} = IPEPS{SquareLattice, T, 5}
function SquareIPEPS(t::AT) where {T,AT<:AbstractArray{T, 5}}
    # NOTE: from here, wrapping `t` with a `IPEPS` type can prevent programing from illegal input with incorrect size.
    size(t,1) == size(t,2) == size(t,3) == size(t,4) || error("size of tensor error, should be `(d, d, d, d, s)`, got $(size(t)).")
    IPEPS{SquareLattice,T,5,AT}(t)
end
getd(ipeps::SquareIPEPS) = size(ipeps.t, 1)
gets(ipeps::SquareIPEPS) = size(ipeps.t, 5)

"""
    indexperm_symmetrize(x::AbstractArray{T,5})

return a normalized `x` whith left-right,
up-down, diagonal and rotational symmetry.
"""
function indexperm_symmetrize(ipeps::SquareIPEPS)
    x = ipeps.t
    x += permutedims(x, (1,4,3,2,5)) # left-right
    x += permutedims(x, (3,2,1,4,5)) # up-down
    x += permutedims(x, (2,1,4,3,5)) # diagonal
    x += permutedims(x, (4,3,2,1,5)) # rotation
    return SquareIPEPS(x / norm(x))
end

@Zygote.adjoint function IPEPS{LT,T,N,AT}(t) where {LT,T,N,AT}
    IPEPS{LT,T,N,AT}(t), dy -> (dy.t,)
end
