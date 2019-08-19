abstract type HamiltonianModel end

struct Ising <: HamiltonianModel end

struct TFIsing{T<:Real} <: HamiltonianModel
    hx::T
end

"""
    hamiltonian(::TFIsing)

return the transverse field ising hamiltonian with transverse magnetic
field `hx` as a two-site operator.
"""
function hamiltonian(model::TFIsing)
    hx = model.hx
    -2 * ein"ij,kl -> ijkl"(σz,σz) -
        hx/2 * ein"ij,kl -> ijkl"(σx, id2) -
        hx/2 * ein"ij,kl -> ijkl"(id2, σx)
end

struct Heisenberg{T<:Real} <: HamiltonianModel
    Jz::T
    Jx::T
    Jy::T
end
Heisenberg() = Heisenberg(1.0,1.0,1.0)

"""
    hamiltonian(::Heisenberg)

return the heisenberg hamiltonian with fields `Jz`, `Jx` and `Jy` as a two-site operator
"""
function hamiltonian(model::Heisenberg)
    h = model.Jz * ein"ij,kl -> ijkl"(σz,σz) -
        model.Jx * ein"ij,kl -> ijkl"(σx, σx) -
        model.Jy * ein"ij,kl -> ijkl"(σy, σy)
    h = ein"ijcd,kc,ld -> ijkl"(h,σx,σx')
    real(h ./ 2)
end
