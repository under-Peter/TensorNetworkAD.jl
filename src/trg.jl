# using OMEinsum
using BackwardsLinalg

@doc raw"
    trg(a, χ, niter)
return the partition-function of a two-dimensional system of size `2^niter`
described by the tensor `a` calculated via the tensor renormalization group
algorithm.
`a` is a rank-4 tensor with the following indices:

        |1
    4--[a]--2
       3|
"
function trg(a::AbstractArray{T,4}, χ, niter) where T
    lnZ = zero(T)
    for n in 1:niter
        maxval = maximum(a)
        a /= maxval
        lnZ += 2^(niter-n+1)*log(maxval)

        dr_ul = einsum("urdl -> drul", (a,))
        ld_ru = einsum("urdl -> ldru", (a,))
        dr, ul = trg_svd(dr_ul, χ)
        ld, ru = trg_svd(ld_ru, χ)

        a = einsum("npu,por,dom,lmn -> urdl", (dr,ld,ul,ru))
    end
    trace = einsum("ijij -> ", (a,))[]
    lnZ += log(trace)
    return lnZ
end


function trg_svd(t, dmax; tol::Float64=1e-12)
    d1, d2, d3, d4 = size(t)
    tmat = reshape(t, d1*d2, d3*d4)
    u, s, v = svd(tmat)
    dmax = min(searchsortedfirst(s, tol, rev=true), dmax)
    FS = s[1:dmax]
    sqrtFSp = sqrt.(FS)
    u = reshape(einsum("ij,j -> ij", (u[:,1:dmax],  sqrtFSp)), (d1, d2, dmax))
    v = reshape(einsum("ij,i -> ij", (copy(v'[1:dmax,:]), sqrtFSp)), (dmax, d3, d4))

    return u, v
end
