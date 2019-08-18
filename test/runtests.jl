using TensorNetworkAD
using Test
using Random

Random.seed!(4)
include("autodiff.jl")
include("exampletensors.jl")
include("fixedpoint.jl")
include("trg.jl")
include("ctmrg.jl")
include("variationalipeps.jl")

@testset "TensorNetworkAD.jl" begin
    println("autodiff tests running...")
    include("autodiff.jl")

    println("exampletensors tests running...")
    include("exampletensors.jl")

    println("fixedpoint tests running...")
    include("fixedpoint.jl")

    println("trg tests running...")
    include("trg.jl")

    println("ctmrg tests running...")
    include("ctmrg.jl")

    println("variationalipeps tests running...")
    include("variationalipeps.jl")
end
