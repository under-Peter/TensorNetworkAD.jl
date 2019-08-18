using Optim, LineSearches
using LinearAlgebra: I, norm

const σx = Float64[0 1; 1 0]
const σy = ComplexF64[0 -1im; 1im 0]
const σz = Float64[1 0; 0 -1]
const id2 = Float64[1 0; 0 1]

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
    energy(h, tin; χ, tol, maxit)

return the energy of the ipeps described by local rank-5 tensors `tin` with
2-site hamiltonian `h` and calculated via a ctmrg with parameters `χ`, `tol`
and `maxit`.
"""
function energy(h::AbstractArray{T,4}, ipeps::IPEPS; χ::Int, tol::Real, maxit::Int) where T
    ipeps = indexperm_symmetrize(ipeps)  # NOTE: this is not good
    D = getd(ipeps)^2
    s = gets(ipeps)
    ap = ein"abcdx,ijkly -> aibjckdlxy"(ipeps.t, conj(ipeps.t))
    ap = reshape(ap, D, D, D, D, s, s)
    a = ein"ijklaa -> ijkl"(ap)

    rt = SquareCTMRGRuntime(a, Val(:raw), χ)
    rt, vals = ctmrg(rt; tol=tol, maxit=maxit)
    e = expectationvalue(h, ap, rt)
    return e
end

"""
    expectationvalue(h, ap, (c,t))

return the expectationvalue of a two-site operator `h` with the sites
described by rank-6 tensor `ap` each and an environment described by
a corner tensor `c` and row/column tensor `t`.
"""
function expectationvalue(h, ap, rt::SquareCTMRGRuntime) where T
    c, t = rt.c, rt.t
    ap /= norm(ap)
    l = ein"ab,ica,bde,cjfdlm,eg,gfk -> ijklm"(c,t,t,ap,c,t)
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
function optimiseipeps(ipeps::IPEPS{LT}, h; χ::Int, tol::Real, maxit::Int,
        optimargs = (),
        optimmethod = LBFGS(m = 20)) where LT
    t = ipeps.t
    let energy = x -> real(energy(h, IPEPS{LT}(x); χ=χ, tol=tol, maxit=maxit))
        res = optimize(energy,
            Δ -> Zygote.gradient(energy,Δ)[1], t, optimmethod, inplace = false, optimargs...)
    end
end
