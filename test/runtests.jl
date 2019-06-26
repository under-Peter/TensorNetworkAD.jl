using TensorNetworkAD
using Test

@testset "TensorNetworkAD.jl" begin
    include("trg.jl")
    include("ctmrg.jl")
    include("variationalipeps.jl")
end
