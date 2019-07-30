using Base.Iterators: drop, take
using IterTools: iterated, imap

function fixedpoint(f, guess, init, stopfun)
    for state in iterated(x -> f(x,init), guess)
        stopfun(state) && return state
    end
end

mutable struct StopFunction{T,S}
    oldvals::T
    counter::Int
    tol::S
    maxit::Int
end


function (st::StopFunction)(state)
    st.counter += 1
    st.counter > st.maxit && return true

    # vals = state[3]
    # diff = norm(vals - st.oldvals)
    # diff <= st.tol && return true
    # st.oldvals = vals

    return false
end

#=
    Following Corboz, Rice and Troyer in **Competing states in the t-J model: uniform d-wave state versus stripe state**
    p.7
=#
using OMEinsum
using BackwardsLinalg
using LinearAlgebra

function initializec(a, χ, randinit)
    c = zeros(eltype(a), χ, χ)
    if randinit
        rand!(c)
        c += adjoint(c)
    else
        cinit = ein"ijkl -> ij"(a)
        foreach(CartesianIndices(cinit)) do i
            i in CartesianIndices(c) && (c[i] = cinit[i])
        end
    end
    return c
end

function initializet(a, χ, randinit)
    t = zeros(eltype(a), χ, size(a,1), χ)
    if randinit
        rand!(t)
        t += permutedims(conj(t), (3,2,1))
    else
        tinit = ein"ijkl -> ijl"(a)
        foreach(CartesianIndices(tinit)) do i
            i in CartesianIndices(t) && (t[i] = tinit[i])
        end
    end
    return t
end

function initializeunitcell(initfun, as, χ, randinit,  T, lvecs)
    uc = UnitCell(Dict{Tuple{Int,Int},T}(), lvecs)
    foreach(u -> uc[u...] = initfun(as[u], χ, randinit), keys(as))
    return uc
end

function initializeunitcells(A, χ, randinit = false)
    lvecs = A.lvecs
    CT = Array{eltype(eltype(A)),2}
    TT = Array{eltype(eltype(A)),3}

    ats = A.tensors
    C1 = initializeunitcell(initializec, ats, χ, randinit, CT, lvecs)
    T1 = initializeunitcell(initializet, ats, χ, randinit, TT, lvecs)

    # rotate counter-clockwise
    ats = Dict(k => ein"ijkl -> lijk"(v) for (k,v) in ats)
    C2 = initializeunitcell(initializec, ats, χ, randinit, CT, lvecs)
    T2 = initializeunitcell(initializet, ats, χ, randinit, TT, lvecs)

    # rotate counter-clockwise
    ats = Dict(k => ein"ijkl -> lijk"(v) for (k,v) in ats)
    C3 = initializeunitcell(initializec, ats, χ, randinit, CT, lvecs)
    T3 = initializeunitcell(initializet, ats, χ, randinit, TT, lvecs)

    # rotate counter-clockwise
    ats = Dict(k => ein"ijkl -> lijk"(v) for (k,v) in ats)
    C4 = initializeunitcell(initializec, ats, χ, randinit, CT, lvecs)
    T4 = initializeunitcell(initializet, ats, χ, randinit, TT, lvecs)

    return C1, C2, C3, C4, T1, T2, T3, T4
end

function nctmrg(a, χ, randinit = false)
    d = size(a[1,1],1)
    C1234T1234 = initializeunitcells(a, χ, randinit)

    oldvals = fill(Inf, 4χ*d)
    tol = 0
    maxit = 10

    stopfun = StopFunction(oldvals, -1, tol, maxit)
    C1234T1234 = nctmrgstep((C1234T1234..., oldvals), (a, χ, d))[1:end-1]
    #C1234T1234 = fixedpoint(nctmrgstep, (C1234T1234..., oldvals), (a, χ, d), stopfun)[1:end-1]
    return C1234T1234
end

function nctmrgstep((C1, C2, C3, C4, T1, T2, T3, T4, oldvals), (a, χ, d))
    C4, T4, C1 = leftmove(C1, C2, C3, C4, T1, T2, T3, T4, a, χ)
    C1, T1, C2 = upmove(C1, C2, C3, C4, T1, T2, T3, T4, a, χ)
    C2, T2, C3 = rightmove(C1, C2, C3, C4, T1, T2, T3, T4, a, χ)
    C3, T3, C4 = downmove(C1, C2, C3, C4, T1, T2, T3, T4, a, χ)

    return (C1, C2, C3, C4, T1, T2, T3, T4, oldvals)
end

function leftmove(C1,C2,C3,C4,T1,T2,T3,T4,a,χ)
    Ps, Pts = similar.((T4,T4))
    C4p, T4p, C1p = similar.((C4, T4, C1))

    foreach(keys(a.tensors)) do k
        Ps[k...], Pts[k...] = leftisommetries(C1,C2,C3,C4,T1,T2,T3,T4,a,χ,k)
    end
    foreach(keys(a.tensors)) do k
        C4p[k...], T4p[k...], C1p[k...] = leftmove(C1,T1,T4,a,C4,T3,Ps,Pts, k)
    end

    return C4p, T4p, C1p
end

function leftmove(C1,T1,T4,A,C4,T3,Ps,Pts, (x,y))
    @ein C1[1,2]   := C1[x,y][-1,-2] * T1[x+1,y][-2,-3,2] * Pts[x,y][-1,-3,1]
    @ein T4[1,2,3] := T4[x,y][-1,-2,-3] * A[x+1,y][-4,2,-5,-2] *
                              Ps[x,y-1][-3,-5,3] * Pts[x,y][-1,-4,1]
    @ein C4[1,2]   := C4[x,y][-1,-2] * T3[x+1,y][1,-3,-1] * Ps[x,y-1][-2,-3,2]
    map(x -> x / norm(x), (C4, T4, C1))
end

function leftisommetries(C1,C2,C3,C4,T1,T2,T3,T4,A,χ,(x,y))
    upperhalf, lowerhalf = horizontalcut(C1,C2,C3,C4,T1,T2,T3,T4,A,(x,y))
    d = size(A[1,1],1)

    lupmat, _ = lq(reshape(upperhalf, d*χ, d*χ))
    ldnmat, _ = lq(reshape(lowerhalf, d*χ, d*χ))
    lup = reshape(lupmat, χ,d,χ*d)
    ldn = reshape(ldnmat, χ,d,χ*d)

    @ein lupldn[1,2] := lupmat[-1,1] * ldnmat[-1,2]

    u, s, vd = svd(lupldn)
    cutoff = min(χ, length(s))
    u = u[:,1:cutoff]
    vd = vd[1:cutoff, :]

    sqrts = sqrt.(pinv.(view(s,1:cutoff)))

    @ein PT[1,2,3] := ldn[1,2,-1] * conj(vd)[3,-1] * sqrts[3]
    @ein P[1,2,3] := lup[1,2,-1] * conj(u)[ -1,3] * sqrts[3]

    return  P, PT
end

function rightmove(C1,C2,C3,C4,T1,T2,T3,T4,a,χ)
    Ps, Pts = similar.((T2,T2))
    C2p, T2p, C3p = similar.((C2, T2, C3))

    foreach(keys(a.tensors)) do k
        Ps[k...], Pts[k...] = rightisommetries(C1,C2,C3,C4,T1,T2,T3,T4,a,χ,k)
    end
    foreach(keys(a.tensors)) do k
        C2p[k...], T2p[k...], C3p[k...] = rightmove(C2,T1,T2,a,C3,T3,Ps,Pts, k)
    end

    return C2p, T2p, C3p
end

function rightmove(C2,T1,T2,A,C3,T3,Ps,Pts, (x,y))
    @ein C2[1,2]   := C2[x,y][-1,-2] * T1[x-1,y][1,-3,-1] * Pts[x,y][-2,-3,2]
    @ein T2[1,2,3] := T2[x,y][-1,-2,-3] * A[x-1,y][-5,-2,-4,2] *
                             Ps[x,y-1][-1,-4,1] * Pts[x,y][-3,-5,3]
    @ein C3[1,2]   := C3[x,y][-1,-2] * T3[x-1,y][-2,-3,2] * Ps[x,y-1][-1,-3,1]
    map(x -> x / norm(x), (C2, T2, C3))
end

function rightisommetries(C1,C2,C3,C4,T1,T2,T3,T4,A,χ,(x,y))
    upperhalf, lowerhalf = horizontalcut(C1,C2,C3,C4,T1,T2,T3,T4,A,(x,y))
    d = size(A[1,1],1)

    _, rupmat = qr(reshape(upperhalf, d*χ, d*χ))
    _, rdnmat = qr(reshape(lowerhalf, d*χ, d*χ))
    rup = reshape(rupmat, χ*d, d, χ)
    rdn = reshape(rdnmat, χ*d, d, χ)

    @ein ruprdn[1,2] :=  rupmat[1,-1] * rdnmat[2,-1]
    u, s, vd = svd(ruprdn)
    cutoff = min(χ, length(s))
    u = u[:,1:cutoff]
    vd = vd[1:cutoff, :]

    sqrts = sqrt.(pinv.(view(s,1:cutoff)))

    @ein PT[1,2,3] := rdn[-1,2,1] * conj(vd)[3,-1] * sqrts[3]
    @ein P[1,2,3] := rup[-1,2,1] * conj(u)[-1,3] * sqrts[3]

    return P, PT
end

function horizontalcut(C1,C2,C3,C4,T1,T2,T3,T4,A,(x,y))
    @ein upperhalf[1,2,3,4] := C1[x  ,y  ][-1,-2]   * T4[x  ,y+1][1,-3,-1] *
                               T1[x+1,y  ][-2,-4,-5] * A[x+1,y+1][2,-6,-4,-3]*
                               T1[x+2,y  ][-5,-7,-8] * A[x+2,y+1][3,-9,-7,-6] *
                               C2[x+3,y  ][-8,-10]  * T2[x+3,y+1][-10,-9,4]

    @ein lowerhalf[1,2,3,4] := C3[x+3,y+3][-1,-2]   * T2[x+3,y+2][4,-3,-1] *
                               T3[x+2,y+3][-2,-4,-5] * A[x+2,y+2][-4,-3,3,-6] *
                               T3[x+1,y+3][-5,-7,-8] * A[x+1,y+2][-7,-6,2,-9] *
                               C4[x  ,y+3][-8,-10]  * T4[x  ,y+2][-10,-9,1]
    return upperhalf, lowerhalf
end

function downmove(C1,C2,C3,C4,T1,T2,T3,T4,a,χ)
    Ps, Pts = similar.((T2,T2))
    C3p, T3p, C4p = similar.((C2, T2, C3))

    foreach(keys(a.tensors)) do k
        Ps[k...], Pts[k...] = downisommetries(C1,C2,C3,C4,T1,T2,T3,T4,a,χ,k)
    end
    foreach(keys(a.tensors)) do k
        C3p[k...], T3p[k...], C4p[k...] = downmove(C3,T2,T3,a,C4,T4,Ps,Pts, k)
    end

    return C3p, T3p, C4p
end

function downisommetries(C1,C2,C3,C4,T1,T2,T3,T4,A,χ,(x,y))
    lefthalf, righthalf = verticalcut(C1,C2,C3,C4,T1,T2,T3,T4,A,(x,y))
    d = size(A[1,1],1)

    llmat, _ = lq(reshape(lefthalf, χ*d, χ*d))
    lrmat, _ = lq(reshape(righthalf, χ*d, χ*d))

    ll = reshape(llmat, χ,d,χ*d)
    lr = reshape(lrmat, χ,d,χ*d)

    @ein lllr[1,2] :=  llmat[-1,1] * lrmat[-1,2]
    u, s, vd = svd(lllr)
    cutoff = min(χ, length(s))
    u = u[:,1:cutoff]
    vd = vd[1:cutoff, :]
    sqrts = sqrt.(pinv.(view(s,1:cutoff)))

    @ein PT[1,2,3] := lr[1,2,-1] * conj(vd)[3,-1] * sqrts[3]
    @ein P[1,2,3] := ll[1,2,-1] * conj(u)[-1,3] * sqrts[3]

    return P, PT
end

function downmove(C3,T2,T3,A,C4,T4,Ps,Pts,(x,y))
    @ein C3[1,2]   := C3[x,y][-1,-2] * T2[x,y-1][1,-3,-1] * Pts[x,y][-2,-3,2]
    @ein T3[1,2,3] := T3[x,y][-1,-2,-3] * A[x-1,y][-2,-4,2,-5] *
                      Ps[x,y-1][-3,-5,3] * Pts[x,y][-1,-4,1]
    @ein C4[1,2]   := C4[x,y][-1,-2] * T4[x-1,y][-2,-3,2] * Ps[x,y-1][-1,-3,1]

    map(x -> x / norm(x), (C3, T3, C4))
end

function upmove(C1,C2,C3,C4,T1,T2,T3,T4,a,χ)
    Ps, Pts = similar.((T1,T1))
    C1p, T1p, C2p = similar.((C1, T1, C2))

    foreach(keys(a.tensors)) do k
        Ps[k...], Pts[k...] = upisommetries(C1,C2,C3,C4,T1,T2,T3,T4,a,χ,k)
    end
    foreach(keys(a.tensors)) do k
        C1p[k...], T1p[k...], C2p[k...] = upmove(C1,T4,T1,a,C2,T2,Ps,Pts, k)
    end

    return C1p, T1p, C2p
end

function upisommetries(C1,C2,C3,C4,T1,T2,T3,T4,A,χ,(x,y))
    lefthalf, righthalf = verticalcut(C1,C2,C3,C4,T1,T2,T3,T4,A,(x,y))
    d = size(A[1,1],1)

    _, rlmat = qr(reshape(lefthalf, χ*d, χ*d))
    _, rrmat = qr(reshape(righthalf, χ*d, χ*d))

    rl = reshape(rlmat, χ*d, d, χ)
    rr = reshape(rrmat, χ*d, d, χ)

    @ein rlrr[1,2] :=  rl[1,-1,-2] * rr[2,-1,-2]
    u, s, vd = svd(rlrr)
    cutoff = min(χ, length(s))
    u, vd = u[:,1:cutoff], vd[1:cutoff, :]
    sqrts = sqrt.(pinv.(view(s,1:cutoff)))

    @ein PT[1,2,3] := rl[-1,2,1] * conj(vd)[3,-1] * sqrts[3]
    @ein P[1,2,3] := rr[-1,2,1] * conj(u)[-1,3] * sqrts[3]

    P, PT
end

function upmove(C1,T4,T1,A,C2,T2,Ps,Pts, (x,y))
    @ein C1[1,2]   := C1[x,y][-1,-2] * T4[x,y][1,-3,-1] * Pts[x,y][-2,-3,2]
    @ein T1[1,2,3] := T1[x,y][-1,-2,-3] * A[x,y][2,-5,-2,-4] *
                      Ps[x,y-1][-1,-4,1] * Pts[x,y][-3,-5,3]
    @ein C2[1,2]   := C2[x,y][-1,-2] * T2[x,y][-2,-3,2] * Ps[x,y-1][-1,-3,1]

    map(x -> x / norm(x), (C1, T1, C2))
end

function verticalcut(C1,C2,C3,C4,T1,T2,T3,T4,A,(x,y))
    @ein lefthalf[1,2,3,4] :=  C4[x  ,y+3][-1,-2]   * T3[x+1,y+3][1,-3,-1] *
                               T4[x  ,y+2][-2,-4,-5] * A[x+1,y+2][-3,2,-6,-4] *
                               T4[x  ,y+1][-5,-7,-8] * A[x+1,y+1][-6,3,-9,-7] *
                               C1[x  ,y  ][-8,-10]  * T1[x+1,y  ][-10,-9,4]
    @ein righthalf[1,2,3,4] := C2[x+3,y  ][-1,-2]   * T1[x+2,y  ][4,-3,-1] *
                               T2[x+3,y+1][-2,-4,-5] * A[x+2,y+1][-6,-4,-3,3] *
                               T2[x+3,y+2][-5,-7,-8] * A[x+2,y+2][-9,-7,-6,2] *
                               C3[x+3,y+3][-8,-10]  * T3[x  ,y+3][-10,-9,1]
    return lefthalf, righthalf
end
