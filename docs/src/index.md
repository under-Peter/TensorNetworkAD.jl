# TensorNetworkAD.jl

This is a package with the goal to implement the algorithms described in [Differentiable Programming Tensor Networks](https://arxiv.org/abs/1903.09650), namely implementing _automatic differentiation_ (AD) on _Corner Transfer Matrix Renormalization Group_ (CTMRG) and _Tensor Renormalization Group_ (TRG),
demonstrating two applications:
- Gradient based optimization of iPEPS
- Direct calculation of energy densities in iPEPS via derivatives of the _free energy_

We aimed for readable and easy to extend code that demonstrates advantages of julia (seamless integration of different packages, performance, readability) and some cutting edge physics.
