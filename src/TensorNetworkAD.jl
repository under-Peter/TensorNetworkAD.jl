module TensorNetworkAD
using Zygote, BackwardsLinalg
using OMEinsum

export trg, num_grad
export ctmrg
export optimiseipeps
export hamiltonian, model_tensor, mag_tensor
export Ising, TFIsing, Heisenberg

include("hamiltonianmodels.jl")

include("trg.jl")
include("fixedpoint.jl")
include("ctmrg.jl")
include("autodiff.jl")
include("variationalipeps.jl")
include("exampletensors.jl")

end # module
