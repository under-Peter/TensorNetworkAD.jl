# Future Directions

To improve upon the package, there are two main places for improvement - `trg` and `ctmrg`.
An area that impacts both is the ability to run on the `GPU`. While a lot of the components are `GPU`-compatible, there are currently problems with `Zygote`. The `add-gpu` branch is a work in progress on compatibility but is hold back by a problem with Zygote at the moment.

## TRG

For `trg`, we should support different lattice-geometries, e.g. triangular or hexagonal. This would require dispatching `trg` on an `AbstractLattice` type and implementing the necessary coarse-graining functions.

Keeping it AD-compatible should be easy since the building blocks remain the same for different geometries.

If Zygote allows, getting the second derivative would allow to the direct calculation of the specific heat.

## CTMRG

`ctmrg` currently only works for tensors that are invariant under permutation of its indices. Adding an implementation that does not assume symmetries would be a big improvement.
This could be combined with the ability to specify a unit-cell for the problem.
First attempts are in the `unitcell`-branch where most of the groundwork is laid but it does not work yet and its interface needs to be updated to the current interface.
 It is based on [this paper](https://arxiv.org/abs/1104.5463).
Again, AD-compatibility should be easy to maintain, given that the code generally uses the same functions.
