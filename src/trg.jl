# using OMEinsum
using BackwardsLinalg

function trg(k, χ, niter)
    D = 2
    inds = 1:D
    M = [sqrt(cosh(k)) sqrt(sinh(k)) ; sqrt(cosh(k)) -sqrt(sinh(k))]
    T = einsum("ij,ik,il,im -> jklm", (M,M,M,M))

    lnZ = zero(k)
    for n in 1:niter
        maxval = maximum(T)
        T /= maxval
        lnZ += 2^(niter-n+1)*log(maxval)

        d2 = size(T,1)^2
        Ma = reshape(einsum("ijkl -> kjil", (T,),), d2, d2)
        Mb = reshape(einsum("ijkl -> lkji", (T,)), d2, d2)
        s1, s3 = trg_svd(Ma, χ)
        s2, s4 = trg_svd(Mb, χ)

        T = einsum("npi,poj,kom,lmn -> ijkl", (s1,s2,s3,s4))
    end
    trace = einsum("ijij -> ", (T,))[]
    lnZ += log(trace)
    return lnZ
end


function trg_svd(Ma, Dmax; tol::Float64=1e-12)
    U, S, V = svd(Ma)
    Dmax = min(searchsortedfirst(S, tol, rev=true), Dmax)
    D = isqrt(size(Ma, 1))
    FS = S[1:Dmax]
    sqrtFSp = sqrt.(FS)
    S1 = reshape(einsum("ij,j -> ij", (U[:,1:Dmax],  sqrtFSp)), (D, D, Dmax))
    S3 = reshape(einsum("ij,i -> ij", (copy(V')[1:Dmax,:], sqrtFSp)), (Dmax, D, D))

    S1, S3
end
