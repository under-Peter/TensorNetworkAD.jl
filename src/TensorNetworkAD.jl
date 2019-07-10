module TensorNetworkAD
using Zygote, BackwardsLinalg
using OMEinsum

export trg, num_grad
export ctmrg
export optimiseipeps
export heisenberghamiltonian, isinghamiltonian, isingtensor

include("trg.jl")
include("fixedpoint.jl")
include("ctmrg.jl")
include("autodiff.jl")
include("variationalipeps.jl")
include("exampletensors.jl")

end # module
