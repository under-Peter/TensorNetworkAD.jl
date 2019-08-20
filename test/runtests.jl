using TensorNetworkAD
using Test
using Random

@testset "TensorNetworkAD.jl" begin
    @testset "autodiff" begin
        println("autodiff tests running...")
        include("autodiff.jl")
    end

    @testset "example tensors" begin
        println("exampletensors tests running...")
        include("exampletensors.jl")
    end

    @testset "fixedpoint" begin
        println("fixedpoint tests running...")
        include("fixedpoint.jl")
    end

    @testset "trg" begin
        println("trg tests running...")
        include("trg.jl")
    end

    @testset "ctmrg" begin
        println("ctmrg tests running...")
        include("ctmrg.jl")
    end

    @testset "variationalipeps" begin
        println("variationalipeps tests running...")
        include("variationalipeps.jl")
    end
end
