using TensorNetworkAD
using Test

@testset "TensorNetworkAD.jl" begin
    include("trg.jl")
    include("ctmrg.jl")
end
