# using OMEinsum
using BackwardsLinalg

function trg(k, χ, niter)
    D = 2
    inds = 1:D
    M = [sqrt(cosh(k)) sqrt(sinh(k)) ; sqrt(cosh(k)) -sqrt(sinh(k))]
    T = meinsum(((-1,1),(-1,2),(-1,3),(-1,4)), (M,M,M,M), (1,2,3,4))

    lnZ = zero(k)
    for n in 1:niter
        maxval = maximum(T)
        T /= maxval
        lnZ += 2^(niter-n+1)*log(maxval)

        d2 = size(T,1)^2
        Ma = reshape(meinsum(((1,2,3,4),), (T,),(3,2,1,4)), d2, d2)
        Mb = reshape(meinsum(((1,2,3,4),), (T,),(4,3,2,1)), d2, d2)
        s1, s3 = trg_svd(Ma, χ)
        s2, s4 = trg_svd(Mb, χ)

        T = meinsum(((-1,-2,1), (-2,-3,2), (3,-3,-4), (4,-4,-1)), (s1,s2,s3,s4),(1,2,3,4))
    end
    trace = meinsum(((1,2,1,2),), (T,), ())[1]
    lnZ += log(trace)
    return lnZ
end


function trg_svd(Ma, Dmax; tol::Float64=1e-12)
    U, S, V = svd(Ma)
    Dmax = min(searchsortedfirst(S, tol, rev=true), Dmax)
    D = isqrt(size(Ma, 1))
    FS = S[1:Dmax]
    sqrtFSp = sqrt.(FS)
    S1 = reshape(meinsum(((1,2),(2,)), (U[:,1:Dmax],  sqrtFSp), (1,2)), (D, D, Dmax))
    S3 = reshape(meinsum(((1,2),(1,)), (copy(V')[1:Dmax,:], sqrtFSp), (1,2)), (Dmax, D, D))

    S1, S3
end
