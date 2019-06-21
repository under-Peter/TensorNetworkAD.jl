module TensorNetworkAD
using Zygote, BackwardsLinalg
using OMEinsum

export trg, num_grad

include("trg.jl")
include("autodiff.jl")

end # module
