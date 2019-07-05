using Optim, LineSearches
using LinearAlgebra: I

const σx = [0 1; 1 0]
const σy = [0 -1im; 1im 0]
const σz = [1 0; 0 -1]
const id2 = [1 0; 0 1]

function rotsymmetrize(x::AbstractArray{<:Any,5})
    (x + permutedims(x, (2,3,4,1,5))
       + permutedims(x, (3,4,1,2,5))
       + permutedims(x, (4,1,2,3,5)))/ (4 * sum(x))
end

function isrotsym(x::AbstractArray{<:Any,5})
    x ≈ permutedims(x, (2,3,4,1,5)) || return false
    x ≈ permutedims(x, (3,4,1,2,5)) || return false
    x ≈ permutedims(x, (4,1,2,3,5)) || return false
    return true
end

function diaglocalhamiltonian(diag::Vector)
    n = length(diag)
    h = einsum("i -> ii", (diag,))
    id = Matrix(I,n,n)
    reshape(h,n,n,1,1) .* reshape(id,1,1,n,n) .+ reshape(h,1,1,n,n) .* reshape(id,n,n,1,1)
end

function tfisinghamiltonian(hx::Float64 = 1.0)
    -2 * einsum("ij,kl -> ijkl", (σz,σz)) -
        hx/2 * einsum("ij,kl -> ijkl", (σx, id2)) -
        hx/2 * einsum("ij,kl -> ijkl", (id2, σx))
end

function heisenberghamiltonian(;Jz::Float64 = 1.0, Jx::Float64 = 1.0, Jy::Float64 = 1.0)
    h = Jz * einsum("ij,kl -> ijkl", (σz,σz)) -
        Jx * einsum("ij,kl -> ijkl", (σx, σx)) -
        Jy * einsum("ij,kl -> ijkl", (σy, σy))
    h = einsum("ijcd,kc,ld -> ijkl", (h,σx,σx'))
    real(h ./ 2)
end

function energy(h, t, χ, tol, maxit)
    t = rotsymmetrize(t)
    d = size(t,1)
    ap = einsum("abcdx,ijkly -> aibjckdlxy", (t, conj(t)))
    ap = reshape(ap, d^2, d^2, d^2, d^2, size(t,5), size(t,5))
    a = einsum("ijklaa -> ijkl", (ap,))
    c, t, vals = ctmrg(a, χ, tol, maxit)

    return expectationvalue(h, ap, (c,t))
end

function expectationvalue(h, a, (c,t))
    id = [1 0; 0 1]
    id2 = reshape(id, 2,2,1,1) .+ reshape(id,1,1,2,2)
    l = einsum("iab,bc,cde,ef,fgk,gdajlm -> ijklm", (t,c,t,c,t,a))
    e = einsum("ijkab,kjicd,abcd -> ", (l,l,h))[]
    n = einsum("ijkll,kjimm -> ", (l,l))[]
    return e/n
end

function optimiseipeps(t, h, χ, tol, maxit;
        optimargs = (),
        optimmethod = LBFGS(m = 20))
    let energy = x -> energy(h, x, χ, tol, maxit)
        res = optimize(energy,
            Δ -> Zygote.gradient(energy,Δ)[1], t, optimmethod, inplace = false, optimargs...)
    end
end
