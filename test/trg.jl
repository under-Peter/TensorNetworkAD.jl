using Zygote
@testset "trg" begin
    foo = x -> trg(x,5,5)
    @test num_grad(foo, 0.4, δ=1e-6) ≈ Zygote.gradient(foo, 0.4)[1]
end
