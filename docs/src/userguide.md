# User Guide

The package provides three algorithms: `trg`, `ctmrg` and `variationalipeps`-optimisation.

## Tensor Renormalization Group

The `trg` function can currently be used to get the partition function of a classical
hamiltonian model on a square lattice.
`trg` uses the principle of _coarse graining_ - low-level detail is discarded
in favor of dynamics that dominate the big picture.
For an excellent guide to implement `trg`, check out [this tutorial for `iTensor`](https://itensor.org/docs.cgi?page=book/trg).

The input to `trg` is a rank-4 tensor `a` that is the tensor-network representation of a model,
a cutoff-dimension `χ` and a number of iterations `niter`.
`trg(a, χ, niter)` returns the partition function _per site_.

The result is per site because otherwise the partition function grows exponentially
(with the system size) in the number of iterations, leading to numerical problems quickly.

The `trg` algorithm is fully differentiable with `Zygote`, which enables us to directly
find the first derivative of the partition function with:
```julia
julia> using Zygote, TensorNetworkAD

julia> Zygote.gradient(0.5) do β
          trg(model_tensor(Ising(), β), 5, 5)
        end
(1.7502426939979507,)
```
where we take the derivative of the ising-partition function w.r.t the inverse
temperature `β` using the `model_tensor` function provided by `TensorNetworkAD`.

## Corner Tensor Renormalization Group

The `ctmrg` function can be used to find a representation of the
environment of an ipeps.
The environment can then be used to calculate local quantities for a system of infinite size such as the magnetisation or (short) correlation-lengths.
For an introduction, I'd recommend  an [overview paper by
Roman Orus](https://arxiv.org/pdf/0905.3225.pdf).

To use the function, first whatever tensor represents the `bulk` of the ipeps
needs to be wrapped in a `CTMRGRuntime` structure which takes care of initializing
the environment - either randomly or from the `bulk` tensor.
Currently, there's only one `CTMRGRuntime` implemented which assumes a bulk-tensor
which is invariant under any permutation of its virtual indices.

The runtime-object can then be provided to `ctmrg` together with a limit to
the number of iterations `maxit` and a tolerance `tol`.
The latter is used to decide convergence - if the sum of absolute differences
in consecutive singular values of the corner is less than `tol`, the algorithm
is converged.

A complete example to get the environment of the Ising model is
```julia
julia> a = model_tensor(Ising(),0.4);
julia> rt = SquareCTMRGRuntime(a, Val(:random), 10);
julia> rt = ctmrg(rt; tol=1e-6, maxit=100);
julia> corner, edge = rt.corner, rt.edge;
```
where `Val(:random)` is used to have the environment initialized
with random values.

## Variational Ipeps Optimisation

Variational Ipeps Optimisation works by combining `ctmrg`, automatic differentiation by `Zygote` and optimisation by `Optim`. The central function is rather simple and can be found in `variationalipeps.jl`.

We provide the function `optimiseipeps` with an `IPEPS`-object - a thin wrapper around a rank-5 tensor - and minimize the energy function with `Optim` using the gradient calculated by `Zygote`.
Energy calculation is built on `ctmrg` so we need to supply its arguments: χ, tol and maxit but we might also modify the optimization algorithm using `optimmethod` or `optimargs`.
`optimargs` can be used to e.g. print out the current energy at each step with `optimargs = (show_trace = true,)`.

The convergence is judged by `Optim` and can be modified with `optimargs`. A complete example looks like:
```julia
julia> using Optim
julia> h = hamiltonian(TFIsing(1.0))
julia> ipeps = SquareIPEPS(randn(2,2,2,2,2))
julia> ipeps = TensorNetworkAD.indexperm_symmetrize(ipeps)
julia> res = optimiseipeps(ipeps, h; χ=5, tol=0, maxit=100,
        optimargs = (Optim.Options(f_tol=1e-6, show_trace=true),))
julia> e = minimum(res)
```
where we get the hamiltonian `h` of the transverse field ising model with magnetic field `hx = 1`,
then we create a random initial `ipeps` with the necessary symmetry
and then minimize its energy with with `optimiseipeps` where we consider it converged if the energy changes by less than `1e-6` between two iterations and we print the energy at each timestep.
The ground-state energy is saved in `e` in the last line.
