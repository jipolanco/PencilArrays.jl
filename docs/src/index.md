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

More generally, PencilArrays can decompose arrays of arbitrary dimension $N$,
along an arbitrary subset of $M$ dimensions.
(In the example above, $N = 3$ and $M = 2$.)

PencilArrays is the basis for the
[PencilFFTs](https://github.com/jipolanco/PencilFFTs.jl) package, which
provides efficient and highly scalable distributed FFTs.

## Features

- distribution of $N$-dimensional arrays among MPI processes;

- decomposition of arrays along an arbitrary subset of dimensions;

- [transpositions](@ref Global-MPI-operations) between different decomposition
  configurations, using point-to-point and collective MPI communications;

- zero-cost, arbitrary dimension permutations à la
  [`PermutedDimsArray`](https://docs.julialang.org/en/latest/base/arrays/#Base.PermutedDimsArrays.PermutedDimsArray);

- convenient [parallel I/O](@ref PencilIO_module) using either MPI-IO or the
  [Parallel HDF5](https://portal.hdfgroup.org/display/HDF5/Parallel+HDF5) libraries;

- distributed FFTs and related transforms via the
  [PencilFFTs](https://github.com/jipolanco/PencilFFTs.jl) package.

## Installation

PencilArrays can be installed using the Julia package manager:

    julia> ] add PencilArrays

## Quick start

The following example assumes that the code is executed on 12 MPI processes.
The processes are distributed on a $3×4$ grid, as in the figure above.

```julia
using MPI
using PencilArrays
using LinearAlgebra: transpose!

MPI.Init()
comm = MPI.COMM_WORLD       # we assume MPI.Comm_size(comm) == 12
rank = MPI.Comm_rank(comm)  # rank of local process, in 0:11

# Define MPI Cartesian topology: distribute processes on a 3×4 grid.
topology = MPITopology(comm, (3, 4))

# Let's decompose 3D arrays along dimensions (2, 3).
# This corresponds to the "x-pencil" configuration in the figure.
# This configuration is described by a Pencil object.
dims_global = (42, 31, 29)  # global dimensions of the array
decomp_dims = (2, 3)
pen_x = Pencil(topology, dims_global, decomp_dims)

# We can now allocate distributed arrays in the x-pencil configuration.
Ax = PencilArray{Float64}(undef, pen_x)
fill!(Ax, rank * π)  # each process locally fills its part of the array
parent(Ax)           # parent array (here, an Array{Float64,3}) holding the local data
size(Ax)             # size of local part
size_global(Ax)      # total size of the array = (42, 31, 29)

# Create another pencil configuration, decomposing along dimensions (1, 3).
# We could use the same constructor as before, but it's recommended to reuse the
# previous Pencil instead.
pen_y = Pencil(pen_x, decomp_dims=(1, 3))

# Now transpose from the x-pencil to the y-pencil configuration, redistributing
# the data initially in Ax.
Ay = PencilArray{Float64}(undef, pen_y)
transpose!(Ay, Ax)
```

[^1]:
    Figure adapted from [this PhD thesis](https://hal.archives-ouvertes.fr/tel-02084215v1).
