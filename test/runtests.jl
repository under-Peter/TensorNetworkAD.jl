using ProgressMeter
pmobj = Progress(29, )
using TensorNetworkAD
using Test
using Random

Random.seed!(4)

@testset "TensorNetworkAD.jl" begin
    include("trg.jl")
    include("ctmrg.jl")
    include("variationalipeps.jl")
end
println()
