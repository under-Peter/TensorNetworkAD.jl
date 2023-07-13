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
    u, s, v = LinearAlgebra.svd(tmat)
    dmax = min(searchsortedfirst(s, tol, rev=true), dmax, length(s))
    FS = s[1:dmax]
    sqrtFSp = sqrt.(FS)
    u = reshape(ein"ij,j -> ij"(u[:,1:dmax],  sqrtFSp), (d1, d2, dmax))
    v = reshape(ein"ij,i -> ij"(copy(v')[1:dmax,:], sqrtFSp), (dmax, d3, d4))

    return u, v
end

struct ZeroAdder end
Base.:+(a, zero::ZeroAdder) = a
Base.:+(zero::ZeroAdder, a) = a
Base.:-(a, zero::ZeroAdder) = a
Base.:-(zero::ZeroAdder, a) = -a
Base.:-(zero::ZeroAdder) = zero

mpow2(a::AbstractArray) = a .^ 2

Zygote.@adjoint function LinearAlgebra.svd(A)
    res = LinearAlgebra.svd(A)
    res, function (dy)
        dU, dS, dVt = dy
        return (svd_back(res.U, res.S, res.V, dU, dS, dVt === nothing ? nothing : dVt'),)
    end
end

"""
    svd_back(U, S, V, dU, dS, dV)

adjoint for SVD decomposition.

References:
    https://j-towns.github.io/papers/svd-derivative.pdf
    https://giggleliu.github.io/2019/04/02/einsumbp.html
"""
function svd_back(U::AbstractArray, S::AbstractArray{T}, V, dU, dS, dV; η::Real=1e-40) where T
    all(x -> x isa Nothing, (dU, dS, dV)) && return nothing
    η = T(η)
    NS = length(S)
    S2 = mpow2(S)
    Sinv = @. S/(S2+η)
    F = S2' .- S2
    F ./= (mpow2(F) .+ η)

    res = ZeroAdder()
    if !(dU isa Nothing)
        UdU = U'*dU
        J = F.*(UdU)
        res += (J+J')*LinearAlgebra.Diagonal(S) + LinearAlgebra.Diagonal(1im*imag(LinearAlgebra.diag(UdU)) .* Sinv)
    end
    if !(dV isa Nothing)
        VdV = V'*dV
        K = F.*(VdV)
        res += LinearAlgebra.Diagonal(S) * (K+K')
    end
    if !(dS isa Nothing)
        res += LinearAlgebra.Diagonal(dS)
    end

    res = U*res*V'

    if !(dU isa Nothing) && size(U, 1) != size(U, 2)
        res += (dU - U* (U'*dU)) * LinearAlgebra.Diagonal(Sinv) * V'
    end

    if !(dV isa Nothing) && size(V, 1) != size(V, 2)
        res = res + U * LinearAlgebra.Diagonal(Sinv) * (dV' - (dV'*V)*V')
    end
    res
end