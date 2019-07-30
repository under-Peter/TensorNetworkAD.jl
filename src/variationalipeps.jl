using Optim, LineSearches
using LinearAlgebra: I, norm

const σx = [0 1; 1 0]
const σy = [0 -1im; 1im 0]
const σz = [1 0; 0 -1]
const id2 = [1 0; 0 1]

function symmetrize(x::AbstractArray{<:Any,5})
    x += permutedims(x, (1,4,3,2,5)) # left-right
    x += permutedims(x, (3,2,1,4,5)) # up-down
    x += permutedims(x, (2,1,4,3,5)) # diagonal
    x += permutedims(x, (4,3,2,1,5)) # rotation
    return x / norm(x)
end

function issym(x::AbstractArray{<:Any,5})
    x ≈ permutedims(x, (2,3,4,1,5)) || return false
    x ≈ permutedims(x, (3,4,1,2,5)) || return false
    x ≈ permutedims(x, (4,1,2,3,5)) || return false
    return true
end

function diaglocalhamiltonian(diag::Vector)
    n = length(diag)
    h = ein"i -> ii"(diag)
    id = Matrix(I,n,n)
    reshape(h,n,n,1,1) .* reshape(id,1,1,n,n) .+ reshape(h,1,1,n,n) .* reshape(id,n,n,1,1)
end

function tfisinghamiltonian(hx::Float64 = 1.0)
    -2 * ein"ij,kl -> ijkl"(σz,σz) -
        hx/2 * ein"ij,kl -> ijkl"(σx, id2) -
        hx/2 * ein"ij,kl -> ijkl"(id2, σx)
end

function heisenberghamiltonian(;Jz::Float64 = 1.0, Jx::Float64 = 1.0, Jy::Float64 = 1.0)
    h = Jz * ein"ij,kl -> ijkl"(σz,σz) -
        Jx * ein"ij,kl -> ijkl"(σx, σx) -
        Jy * ein"ij,kl -> ijkl"(σy, σy)
    h = ein"ijcd,kc,ld -> ijkl"(h,σx,σx')
    real(h ./ 2)
end

function energy(h, tin, χ, tol, maxit)
    tin = symmetrize(tin)
    d = size(tin,1)
    ap = ein"abcdx,ijkly -> aibjckdlxy"(tin, conj(tin))
    ap = reshape(ap, d^2, d^2, d^2, d^2, size(tin,5), size(tin,5))
    a = ein"ijklaa -> ijkl"(ap)
    c, t = ctmrg(a, χ, tol, maxit)
    e = expectationvalue(h, ap, (c,t))

    return e
end

function expectationvalue(h, a, (c,t))
    a /= norm(a)
    l = ein"ab,ica,bde,cjfdlm,eg,gfk -> ijklm"(c,t,t,a,c,t)
    e = ein"abcij,abckl,ijkl -> "(l,l,h)[]
    n = ein"ijkaa,ijkbb -> "(l,l)[]
    return e/n
end

function optimiseipeps(t, h, χ, tol, maxit;
        optimargs = (),
        optimmethod = LBFGS(m = 20))
    let energy = x -> real(energy(h, x, χ, tol, maxit))
        res = optimize(energy,
            Δ -> Zygote.gradient(energy,Δ)[1], t, optimmethod, inplace = false, optimargs...)
    end
end
