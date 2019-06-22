# tensor for classical 2-d model
const isingβc = log(1+sqrt(2))/2

function tensorfromclassical(ham::Matrix)
    wboltzmann = exp.(ham)
    q = sqrt(wboltzmann)
    einsum("ij,ik,il,im -> jklm", (q,q,q,q))
end

function isingtensor(β)
    a = reshape(Float64[1 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 1], 2,2,2,2)
    cβ, sβ = sqrt(cosh(β)), sqrt(sinh(β))
    q = 1/sqrt(2) * [cβ+sβ cβ-sβ; cβ-sβ cβ+sβ]
    einsum("abcd,ai,bj,ck,dl -> ijkl", (a,q,q,q,q))
end

function isingmagtensor(β)
    a = reshape([1 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 -1] , 2,2,2,2)
    cβ, sβ = sqrt(cosh(β)), sqrt(sinh(β))
    q = 1/sqrt(2) * [cβ+sβ cβ-sβ; cβ-sβ cβ+sβ]
    einsum("abcd,ai,bj,ck,dl -> ijkl", (a,q,q,q,q))
end

function magnetisationofβ(β, χ)
    a = isingtensor(β)
    m = isingmagtensor(β)
    c, t, = TensorNetworkAD.ctmrg(a, χ, 1e-6, 100)
    ctc  = einsum("ia,ajb,bk -> ijk", (c,t,c))
    env  = einsum("alc,ckd,bjd,bia -> ijkl",(ctc,t,ctc,t))
    mag  = einsum("ijkl,ijkl ->", (env,m))[]
    norm = einsum("ijkl,ijkl ->", (env,a))[]

    return abs(mag/norm)
end

function eoeofβ(β,χ)
    a = isingtensor(β)
    m = isingmagtensor(β)
    c, t, = ctmrg(a, χ, 1e-6, 100)
    d = diag(c)
    d = d ./ norm(d,1)
    return -sum(d .* log2.(d))
end

magofβ(β) = β > isingβc ? (1-sinh(2*β)^-4)^(1/8) : 0.
