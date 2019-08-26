<!-- # TensorNetworkAD -->
<div align="center"> <img
src="tnad-logo.png"
alt="TensorNetworkAD logo" width="510"></img>
<h1>TensorNetworkAD</h1>
</div>

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://under-Peter.github.io/TensorNetworkAD.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://under-Peter.github.io/TensorNetworkAD.jl/dev)
[![Build Status](https://travis-ci.com/under-Peter/TensorNetworkAD.jl.svg?branch=master)](https://travis-ci.com/under-Peter/TensorNetworkAD.jl)
[![Codecov](https://codecov.io/gh/under-Peter/TensorNetworkAD.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/under-Peter/TensorNetworkAD.jl)



This is a repository for the _Google Summer of Code_ project on Differentiable Tensor Networks.

In this package we implemented the algorithms described in [Differentiable Programming Tensor Networks](https://arxiv.org/abs/1903.09650), namely implementing _automatic differentiation_ (AD) on _Corner Transfer Matrix Renormalization Group_ (CTMRG) and _Tensor Renormalization Group_ (TRG),
demonstrating two applications:
- Gradient based optimization of iPEPS
- Direct calculation of energy densities in iPEPS via derivatives of the _free energy_

More generally we aim to provide Julia with the tools to combine AD and tensor network methods.


Suggestions and Comments in the _Issues_ are welcome.

## Example

Since this package was inspired by the
[Differentiable Programming Tensor Networks](https://arxiv.org/abs/1903.09650)
paper, we demonstrate the usage of those algorithms in the following.

### Free Energy of the 2D Classical Ising Model

We start by constructing the tensor for the tensor network representation of the 2d classical Ising Model.
This tensor can be constructed using the `model_tensor`-function that takes a `model`-parameter - in our case `Ising()` - and an inverse temperature `β` (e.g. at `β=0.5`).
```julia
julia> a = model_tensor(Ising(), 0.5)
2×2×2×2 Array{Float64,4}:
[:, :, 1, 1] =
 2.53434  0.5    
 0.5      0.18394

[:, :, 2, 1] =
 0.5      0.18394
 0.18394  0.5    

[:, :, 1, 2] =
 0.5      0.18394
 0.18394  0.5    

[:, :, 2, 2] =
 0.18394  0.5    
 0.5      2.53434
```

Using the `trg` function, we can calculate the partition function of the model per site:
```julia
julia> trg(a, 20,20)
1.0257933734351765
```
which grows bond-dimensions up to `20` and does `20` iterations, i.e. grows the system to a size of `2^20` which is well converged for our purposes.

Given the partition function, we get the free energy as the first derivative with respect to `β` times `-1`.
 With Zygote, this is straightforward to calculate:
```julia
julia> using Zygote: gradient

julia> dβ = gradient(β -> trg(model_tensor(Ising(), β), 20,20), 0.5)[1]
1.7455677143228514

julia> -dβ
-1.7455677143228514
```
which agrees with the data presented in the paper.

### Finding the Ground State of infinite 2D Heisenberg model

The other algorithm variationally minimizes the energy of a Heisenberg model on a two-dimensional infinite lattice using a form of gradient descent.

First, we need the hamiltonian as a tensor network operator
```
julia> h = hamiltonian(Heisenberg())
2×2×2×2 Array{Float64,4}:
[:, :, 1, 1] =
 -0.5  0.0
  0.0  0.5

[:, :, 2, 1] =
  0.0  0.0
 -1.0  0.0

[:, :, 1, 2] =
 0.0  -1.0
 0.0   0.0

[:, :, 2, 2] =
 0.5   0.0
 0.0  -0.5
```
where we get the `Heisenberg`-hamiltonian with default parameters `Jx = Jy = Jz = 1.0`.
Next we initialize an ipeps-tensor and calculate the energy of that tensor and the hamiltonian:
```julia
julia> ipeps = SquareIPEPS(rand(2,2,2,2,2));

julia> ipeps = TensorNetworkAD.indexperm_symmetrize(ipeps);

julia> TensorNetworkAD.energy(h,ipeps, χ=20, tol=1e-6,maxit=100)
-0.5278485155836766
```
where the initial energy is random.

To minimise it, we combine `Optim` and `Zygote` under the hood to provide the `optimiseipeps` function.
```julia
julia> using Optim

julia> res = optimiseipeps(ipeps, h; χ=20, tol=1e-6, maxit=100,
        optimargs = (Optim.Options(f_tol=1e-6, show_trace=true),));
Iter     Function value   Gradient norm
     0    -5.015158e-01     2.563357e-02
 * time: 4.100799560546875e-5
     1    -6.171409e-01     3.170732e-02
 * time: 0.3943500518798828
     2    -6.558814e-01     2.927539e-02
 * time: 0.6722378730773926
     3    -6.577320e-01     1.299056e-02
 * time: 1.0529990196228027
     4    -6.587514e-01     8.515789e-03
 * time: 1.2889769077301025
     5    -6.595896e-01     1.102446e-02
 * time: 1.5059330463409424
     6    -6.599735e-01     2.020418e-03
 * time: 1.8917429447174072
     7    -6.600449e-01     4.343536e-03
 * time: 2.180701971054077
     8    -6.601202e-01     2.623793e-03
 * time: 2.5907390117645264
     9    -6.602188e-01     3.951503e-04
 * time: 2.9895379543304443
    10    -6.602232e-01     2.597750e-04
 * time: 3.254667043685913
    11    -6.602246e-01     2.960359e-04
 * time: 3.4899749755859375
    12    -6.602282e-01     2.846450e-04
 * time: 3.739893913269043
    13    -6.602290e-01     1.679273e-04
 * time: 3.9142658710479736
    14    -6.602303e-01     2.155790e-04
 * time: 4.230381011962891
    15    -6.602311e-01     2.239934e-05
 * time: 4.5699989795684814
    16    -6.602311e-01     1.935087e-05
 * time: 4.837096929550171
```
where our final value for the energy `e = -0.6602` agrees with the value found in the paper.
## License
MIT License
