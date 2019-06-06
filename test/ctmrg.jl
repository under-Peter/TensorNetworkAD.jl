using TensorNetworkAD
using Test, Random
using Zygote

@testset "ctmrg" begin
    Random.seed!(2)
    χ = 5
    niter = 5
    # $ python 2_variational_iPEPS/variational.py -model TFIM -D 3 -chi 10 -Niter 10 -Nepochs 10
    @test_broken isapprox(ctmrg(:TFIM; nepochs=10, χ=10, D=3, niter=10).E, -2.1256619, atol=1e-5)
end
