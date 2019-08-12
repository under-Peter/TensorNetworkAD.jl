using TensorNetworkAD
using Test
using Random

Random.seed!(4)

@testset "TensorNetworkAD.jl" begin
    println("trg tests running...")
    include("trg.jl")
    println("ctmrg tests running...")
    include("ctmrg.jl")
    println("variationalipeps tests running...")
    include("variationalipeps.jl")
end
