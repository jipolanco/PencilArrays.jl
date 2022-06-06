```@meta
CurrentModule = PencilArrays
```

# PencilArrays

Distributed Julia arrays using the MPI protocol.

## Introduction

This package provides a convenient framework for working with multidimensional
Julia arrays distributed among MPI processes.

The name of this package originates from the decomposition of 3D domains along
two out of three dimensions, sometimes called *pencil* decomposition.
This is illustrated by the figure below,[^1] where each coloured block is
managed by a different MPI process.

```@raw html
<div class="figure">
  <img
    width="85%"
    src="img/pencils.svg"
    alt="Pencil decomposition of 3D domains">
</div>
```

More generally, PencilArrays can decompose arrays of arbitrary dimension ``N``,
along an arbitrary subset of ``M`` dimensions.
(In the example above, ``N = 3`` and ``M = 2``.)

PencilArrays is the basis for the
[PencilFFTs](https://github.com/jipolanco/PencilFFTs.jl) package, which
provides efficient and highly scalable distributed FFTs.

## Features

- distribution of ``N``-dimensional arrays among MPI processes;

- decomposition of arrays along an arbitrary subset of dimensions;

- tools for conveniently and efficiently iterating over the [coordinates of
  distributed multidimensional geometries](@ref Working-with-grids);

- [transpositions](@ref Global-MPI-operations) between different decomposition
  configurations, using point-to-point and collective MPI communications;

- zero-cost, convenient dimension permutations using [StaticPermutations.jl](https://github.com/jipolanco/StaticPermutations.jl);

- convenient [parallel I/O](@ref PencilIO_module) of distributed arrays using
  either MPI-IO or the [Parallel
  HDF5](https://portal.hdfgroup.org/display/HDF5/Parallel+HDF5) libraries;

- distributed FFTs and related transforms via the
  [PencilFFTs.jl](https://github.com/jipolanco/PencilFFTs.jl) package.

## Installation

PencilArrays can be installed using the Julia package manager:

    julia> ] add PencilArrays

## Quick start

```julia
using MPI
using PencilArrays

MPI.Init()
comm = MPI.COMM_WORLD       # MPI communicator
rank = MPI.Comm_rank(comm)  # rank of local process

# Let's decompose a 3D grid across all MPI processes.
# The resulting configuration is described by a Pencil object.
dims_global = (42, 31, 29)  # global dimensions of the array
pen_x = Pencil(dims_global, comm)

# By default the 3D grid is decomposed along the two last dimensions, similarly
# to the "x-pencil" configuration in the figure above:
println(pen_x)
# Decomposition of 3D data
#   Data dimensions: (42, 31, 29)
#   Decomposed dimensions: (2, 3)
#   Data permutation: NoPermutation()
#   Array type: Array

# We can now allocate distributed arrays in the x-pencil configuration.
Ax = PencilArray{Float64}(undef, pen_x)
fill!(Ax, rank * π)  # each process locally fills its part of the array
parent(Ax)           # parent array holding the local data (here, an Array{Float64,3})
size(Ax)             # total size of the array = (42, 31, 29)
size_local(Ax)       # size of local part, e.g. (42, 8, 10) for a given process
range_local(Ax)      # range of local part on global grid, e.g. (1:42, 16:23, 20:29)

# Let's associate the dimensions to a global grid of coordinates (x_i, y_j, z_k)
xs_global = range(0, 1;  length = dims_global[1])
ys_global = range(0, 2;  length = dims_global[2])
zs_global = range(0, 2π; length = dims_global[3])

# Part of the grid associated to the local MPI process:
grid = localgrid(pen_x, (xs_global, ys_global, zs_global))

# This is convenient for example if we want to initialise the `Ax` array as
# a function of the grid coordinates (x, y, z):
@. Ax = grid.x + (2 * grid.y * cos(grid.z))

# Alternatively (useful in higher dimensions):
@. Ax = grid[1] + (2 * grid[2] * cos(grid[3]))

# Create another pencil configuration, decomposing along dimensions (1, 3).
# We could use the same constructor as before, but it's recommended to reuse the
# previous Pencil instead to reduce memory usage.
pen_y = Pencil(pen_x; decomp_dims = (1, 3))

# Now transpose from the x-pencil to the y-pencil configuration, redistributing
# the data initially in Ax.
Ay = PencilArray{Float64}(undef, pen_y)
transpose!(Ay, Ax)

# We can check that Ax and Ay have the same data (but distributed differently)
# by combining the data from all different processes onto a single process
# (this should never be used for large datasets!)
gather(Ax) == gather(Ay)  # true
```

[^1]:
    Figure adapted from [this PhD thesis](https://hal.archives-ouvertes.fr/tel-02084215v1).
