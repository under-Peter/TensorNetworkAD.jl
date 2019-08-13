using Zygote
using LinearAlgebra

@Zygote.nograd StopFunction

# patch
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
"
num_grad(f, K::Real; δ::Real = 1e-5) = (f(K+δ/2) - f(K-δ/2))/δ

@doc raw"
    num_grad(f, K::AbstractArray; [δ = 1e-5])
return the numerical gradient of `f` for each element of `K`.
"
function num_grad(f, a::AbstractArray; δ::Real = 1e-5)
    map(CartesianIndices(a)) do i
        foo = x -> (ac = copy(a); ac[i] = x; f(ac))
        num_grad(foo, a[i], δ = δ)
    end
end
