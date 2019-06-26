using TensorNetworkAD
using Test
using ProgressMeter
p = ProgressUnknown("Test run:")

@testset "TensorNetworkAD.jl" begin
    next!(p)
    include("trg.jl")
    next!(p)
    include("ctmrg.jl")
    next!(p)
    include("variationalipeps.jl")
    finish!(p)
end
