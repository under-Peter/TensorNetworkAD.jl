module TensorNetworkAD
using Zygote, LinalgBackwards
using OMEinsum

export trg, num_grad

include("einsum.jl")
include("trg.jl")
include("autodiff.jl")

end # module
