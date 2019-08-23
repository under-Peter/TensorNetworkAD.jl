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
function trg(a::AbstractArray{T,4}, χ, niter; tol::Float64 = 1e-16) where T
    lnZ = zero(T)
    for n in 1:niter
        maxval = maximum(abs.(a))
        a /= maxval
        lnZ += 2.0^(1-n)*log(maxval)

        dr_ul = ein"urdl -> drul"(a)
        ld_ru = ein"urdl -> ldru"(a)
        dr, ul = trg_svd(dr_ul, χ, tol)
        ld, ru = trg_svd(ld_ru, χ, tol)

        a = ein"npu,por,dom,lmn -> urdl"(dr,ld,ul,ru)
    end
    trace = ein"ijij -> "(a)[]
    lnZ += log(trace)/2.0^niter
    return lnZ
end


function trg_svd(t, dmax, tol)
    d1, d2, d3, d4 = size(t)
    tmat = reshape(t, d1*d2, d3*d4)
    u, s, v = svd(tmat)
    dmax = min(searchsortedfirst(s, tol, rev=true), dmax, length(s))
    FS = s[1:dmax]
    sqrtFSp = sqrt.(FS)
    u = reshape(ein"ij,j -> ij"(u[:,1:dmax],  sqrtFSp), (d1, d2, dmax))
    v = reshape(ein"ij,i -> ij"(copy(v')[1:dmax,:], sqrtFSp), (dmax, d3, d4))

    return u, v
end
