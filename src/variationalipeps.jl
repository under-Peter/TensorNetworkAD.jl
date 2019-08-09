using Optim, LineSearches
using LinearAlgebra: I, norm

const σx = Float64[0 1; 1 0]
const σy = ComplexF64[0 -1im; 1im 0]
const σz = Float64[1 0; 0 -1]
const id2 = Float64[1 0; 0 1]

"""
    symmetrize(x::AbstractArray{T,5})

return a normalized `x` whith left-right,
up-down, diagonal and rotational symmetry.
"""
function symmetrize(x::AbstractArray{<:Any,5})
    x += permutedims(x, (1,4,3,2,5)) # left-right
    x += permutedims(x, (3,2,1,4,5)) # up-down
    x += permutedims(x, (2,1,4,3,5)) # diagonal
    x += permutedims(x, (4,3,2,1,5)) # rotation
    return x / norm(x)
end

"""
    diaglocalhamiltonian(diag::Vector)
return the 2-site Hamiltonian with single-body terms given
by the diagonal `diag`.
"""
function diaglocalhamiltonian(diag::Vector)
    n = length(diag)
    h = ein"i -> ii"(diag)
    id = Matrix(I,n,n)
    reshape(h,n,n,1,1) .* reshape(id,1,1,n,n) .+ reshape(h,1,1,n,n) .* reshape(id,n,n,1,1)
end

"""
    tfisinghamiltonian(hx::Float64 = 1.0)

return the transverse field ising hamiltonian with transverse magnetic
field `hx` as a two-site operator.
"""
function tfisinghamiltonian(hx::Float64 = 1.0)
    -2 * ein"ij,kl -> ijkl"(σz,σz) -
        hx/2 * ein"ij,kl -> ijkl"(σx, id2) -
        hx/2 * ein"ij,kl -> ijkl"(id2, σx)
end

"""
    heisenberghamiltonian(;Jz::Float64 = 1.0, Jx::Float64 = 1.0, Jy::Float64 = 1.0)

return the heisenberg hamiltonian with fields `Jz`, `Jx` and `Jy` as a two-site operator
"""
function heisenberghamiltonian(;Jz::Float64 = 1.0, Jx::Float64 = 1.0, Jy::Float64 = 1.0)
    h = Jz * ein"ij,kl -> ijkl"(σz,σz) -
        Jx * ein"ij,kl -> ijkl"(σx, σx) -
        Jy * ein"ij,kl -> ijkl"(σy, σy)
    h = ein"ijcd,kc,ld -> ijkl"(h,σx,σx')
    real(h ./ 2)
end

"""
    energy(h, tin, χ, tol, maxit)

return the energy of the ipeps described by local rank-5 tensors `tin` with
2-site hamiltonian `h` and calculated via a ctmrg with parameters `χ`, `tol`
and `maxit`.
"""
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

"""
    expectationvalue(h, a, (c,t))

return the expectationvalue of a two-site operator `h` with the sites
described by rank-5 tensor `a` each and an environment described by
a corner tensor `c` and row/column tensor `t`.
"""
function expectationvalue(h, a, (c,t))
    a /= norm(a)
    l = ein"ab,ica,bde,cjfdlm,eg,gfk -> ijklm"(c,t,t,a,c,t)
    e = ein"abcij,abckl,ijkl -> "(l,l,h)[]
    n = ein"ijkaa,ijkbb -> "(l,l)[]
    return e/n
end

"""
    optimiseipeps(t, h, χ, tol, maxit; optimargs = (), optimmethod = LBFGS(m = 20))

return the tensor `t'` that describes an ipeps that minimises the energy of the
two-site hamiltonian `h`. The minimization is done using `Optim` with default-method
`LBFGS`. Alternative methods can be specified by loading `LineSearches` and
providing `optimmethod`. Other options to optim can be passed with `optimargs`.
The energy is calculated using ctmrg with parameters `χ`, `tol` and `maxit`.
"""
function optimiseipeps(t, h, χ, tol, maxit;
        optimargs = (),
        optimmethod = LBFGS(m = 20))
    let energy = x -> real(energy(h, x, χ, tol, maxit))
        res = optimize(energy,
            Δ -> Zygote.gradient(energy,Δ)[1], t, optimmethod, inplace = false, optimargs...)
    end
end
