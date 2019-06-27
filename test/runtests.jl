using TensorNetworkAD
using Test
using ProgressMeter
using Random
p = ProgressUnknown("Test run:")

Random.seed!(4)

@testset "TensorNetworkAD.jl" begin
    next!(p)
    include("trg.jl")
    next!(p)
    include("ctmrg.jl")
    next!(p)
    include("variationalipeps.jl")
    finish!(p)
end
