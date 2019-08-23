export IPEPS, SquareIPEPS
"""
    IPEPS{LT<:AbstractLattice, T, N}

Infinite projected entangled pair of states.
`LT` is the type of lattice, `T` and `N` are bulk tensor element type and order.
"""
struct IPEPS{LT<:AbstractLattice, T, N, AT<:AbstractArray{T, N}}
    bulk::AT
    # TODO: check input size in constructor
end
IPEPS{LT}(bulk::AT) where {LT,T,N,AT<:AbstractArray{T,N}} = IPEPS{LT,T,N,AT}(bulk)

############### IPEPS on square lattice ###################
# size of bulk is `d × d × d × d × s`
const SquareIPEPS{T} = IPEPS{SquareLattice, T, 5}
function SquareIPEPS(bulk::AT) where {T,AT<:AbstractArray{T, 5}}
    # NOTE: from here, wrapping `bulk` with a `IPEPS` type can prevent programing from illegal input with incorrect size.
    size(bulk,1) == size(bulk,2) == size(bulk,3) == size(bulk,4) || throw(DimensionMismatch(
        "size of tensor error, should be `(d, d, d, d, s)`, got $(size(bulk))."))
    IPEPS{SquareLattice,T,5,AT}(bulk)
end
getd(ipeps::SquareIPEPS) = size(ipeps.bulk, 1)
gets(ipeps::SquareIPEPS) = size(ipeps.bulk, 5)

"""
    indexperm_symmetrize(ipeps::SquareIPEPS)

return a `SquareIPEPS` based on `ipeps` that is symmetric under
permutation of its virtual indices.
"""
function indexperm_symmetrize(ipeps::SquareIPEPS)
    x = ipeps.bulk
    x += permutedims(x, (1,4,3,2,5)) # left-right
    x += permutedims(x, (3,2,1,4,5)) # up-down
    x += permutedims(x, (2,1,4,3,5)) # diagonal
    x += permutedims(x, (4,3,2,1,5)) # rotation
    return SquareIPEPS(x / norm(x))
end
