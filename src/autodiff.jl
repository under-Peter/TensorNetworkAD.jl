using Zygote
using LinearAlgebra

@Zygote.nograd StopFunction
@Zygote.nograd _initializect_square

@Zygote.adjoint function IPEPS{LT,T,N,AT}(bulk) where {LT,T,N,AT}
    IPEPS{LT,T,N,AT}(bulk), dy -> (dy.bulk,)
end

@Zygote.adjoint function CTMRGRuntime{LT}(bulk::AT,
        corner::AbstractArray{T}, edge::AbstractArray{T}) where {LT<:AbstractLattice,T,N,AT<:AbstractArray{T,N}}
        return CTMRGRuntime{LT}(bulk,corner,edge), dy->(dy.bulk, dy.corner, dy.edge)
end

# patch since it's currently broken otherwise
@Zygote.adjoint function Base.typed_hvcat(::Type{T}, rows::Tuple{Vararg{Int}}, xs::S...) where {T,S}
  Base.typed_hvcat(T,rows, xs...), ȳ -> (nothing, nothing, permutedims(ȳ)...)
end

# improves performance compared to default implementation, also avoids errors
# with some complex arrays
@Zygote.adjoint function LinearAlgebra.norm(A::AbstractArray, p::Real = 2)
    n = norm(A,p)
    back(Δ) = let n = n
                    (Δ .* A ./ (n + eps(0f0)),)
                end
    return n, back
end

@doc raw"
    num_grad(f, K::Real; [δ = 1e-5])

return the numerical gradient of `f` at `K` calculated with
`(f(K+δ/2) - f(K-δ/2))/δ`

# example

```jldoctest; setup = :(using TensorNetworkAD)
julia> TensorNetworkAD.num_grad(x -> x * x, 3) ≈ 6
true
```
"
num_grad(f, K::Real; δ::Real = 1e-5) = (f(K+δ/2) - f(K-δ/2))/δ

@doc raw"
    num_grad(f, K::AbstractArray; [δ = 1e-5])
return the numerical gradient of `f` for each element of `K`.

# example

```jldoctest; setup = :(using TensorNetworkAD, LinearAlgebra)
julia> TensorNetworkAD.num_grad(tr, rand(2,2)) ≈ I
true
```
"
function num_grad(f, a::AbstractArray; δ::Real = 1e-5)
    map(CartesianIndices(a)) do i
        foo = x -> (ac = copy(a); ac[i] = x; f(ac))
        num_grad(foo, a[i], δ = δ)
    end
end
