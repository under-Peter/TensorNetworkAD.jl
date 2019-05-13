<!-- # TensorNetworkAD -->
<div align="center"> <img
src="tnad-logo.png"
alt="TensorNetworkAD logo" width="510"></img>
<h1>TensorNetworkAD</h1>
</div>

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://under-Peter.github.io/TensorNetworkAD.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://under-Peter.github.io/TensorNetworkAD.jl/dev)
[![Build Status](https://travis-ci.com/under-Peter/TensorNetworkAD.jl.svg?branch=master)](https://travis-ci.com/under-Peter/TensorNetworkAD.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/under-Peter/TensorNetworkAD.jl?svg=true)](https://ci.appveyor.com/project/under-Peter/TensorNetworkAD-jl)
[![Codecov](https://codecov.io/gh/under-Peter/TensorNetworkAD.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/under-Peter/TensorNetworkAD.jl)



This is a repository for the _Google Summer of Code_ project on Differentiable Tensor Networks.
It is a work in progress and will **change substantially this summer (2019)** - no guarantees can be made.

The goal is to implement the algorithms described in [Differentiable Programming Tensor Networks](https://arxiv.org/abs/1903.09650), namely implementing _automatic differentiation_ (AD) on _Corner Transfer Matrix Renormalization Group_ (CTMRG) and _Tensor Renormalization Group_ (TRG),
demonstrating two applications:
- Gradient based optimization of iPEPS
- Direct calculation of energy densities in iPEPS via derivatives of the _free energy_

More generally we aim to provide Julia with the tools to combine AD and tensor network methods.


Suggestions and Comments in the _Issues_ are welcome.

## License
MIT License
