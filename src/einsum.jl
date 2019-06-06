using OMEinsum
using TupleTools
@doc raw"
    contractionindices(ixs, iy)
calculate the indices of the intermediate tensor
resulting from contracting the first two indices of
`ixs`.
returns both the indices of the contraction and the intermediate
result indices.

# Example

Product of three matrices
```julia
julia> contractionindices(((1,2),(2,3),(3,4)), (1,4))
(((1, 2), (2, 3)), (1, 3))
```
"
function contractionindices(ixs::NTuple{N, T where T}, iy) where N
    N <= 2 && return (ixs, iy)
    ix12 = unique(vcat(collect(ixs[1]), collect(ixs[2])))
    allothers = vcat(map(collect,ixs[3:end])..., collect(iy))
    ((ixs[1], ixs[2]), Tuple(i for i in ix12 if i in allothers))
end

@doc raw"
    meinsum(ixs, xs, iy)
like `einsum(ixs,xs,iy)` but contracting pairwise from
left to right.
"
function meinsum(ixs, xs, iy)
    ixstmp, iytmp = contractionindices(ixs, iy)
    xstmp = Tuple(xs[i] for i in 1:length(ixstmp))
    x = einsum(ixstmp, xstmp, iytmp)
    length(xs) <= 2 && return x
    nixs = (iytmp, map(i -> ixs[i], (3:length(ixs)...,))...)
    nxs = (x, map(i -> xs[i], (3:length(ixs)...,))...)
    return meinsum(nixs, nxs, iy)
end
