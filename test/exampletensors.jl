using TensorNetworkAD
using Test
using Zygote, OMEinsum
using TensorNetworkAD: model_tensor, tensorfromclassical, trg_svd

@testset "exampletensor" begin
    β = rand()
    @test model_tensor(Ising(),β) ≈ tensorfromclassical([β -β; -β β])
end
