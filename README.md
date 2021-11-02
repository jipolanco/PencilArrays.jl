# PencilArrays

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jipolanco.github.io/PencilArrays.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jipolanco.github.io/PencilArrays.jl/dev)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5148035.svg)](https://doi.org/10.5281/zenodo.5148035)

[![Build Status](https://github.com/jipolanco/PencilArrays.jl/workflows/CI/badge.svg)](https://github.com/jipolanco/PencilArrays.jl/actions)
[![Coverage](https://codecov.io/gh/jipolanco/PencilArrays.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jipolanco/PencilArrays.jl)

Distributed Julia arrays using the MPI protocol.

This package provides a convenient framework for working with multidimensional
Julia arrays distributed among MPI processes.

The name of this package originates from the decomposition of 3D domains along
two out of three dimensions, sometimes called *pencil* decomposition.
This is illustrated by the figure below, which represents a distributed 3D array.
Each coloured block is managed by a different MPI process.

<p align="center">
  <br/>
  <img width="85%" alt="Pencil decomposition of 3D domains" src="docs/src/img/pencils.svg">
</p>

More generally, PencilArrays can decompose arrays of arbitrary dimension `N`,
along an arbitrary number of subdimensions `M < N`.
(In the image above, `N = 3` and `M = 2`.)

PencilArrays is the basis for the
[PencilFFTs](https://github.com/jipolanco/PencilFFTs.jl) package, which
provides efficient and highly scalable distributed FFTs.

## Features

- distribution of `N`-dimensional arrays among MPI processes;

- decomposition of arrays along an arbitrary subset of dimensions;

- transpositions between different decomposition configurations, using
  point-to-point and collective MPI communications;

- zero-cost, convenient dimension permutations using [StaticPermutations](https://github.com/jipolanco/StaticPermutations.jl);

- convenient parallel I/O using either MPI-IO or the [Parallel
  HDF5](https://portal.hdfgroup.org/display/HDF5/Parallel+HDF5) libraries;

- distributed FFTs and related transforms via the
  [PencilFFTs](https://github.com/jipolanco/PencilFFTs.jl) package.

## Installation

PencilArrays can be installed using the Julia package manager:

    julia> ] add PencilArrays

## Quick start

The following example assumes that the code is executed on 12 MPI processes.
The processes are distributed on a 3×4 grid, as in the figure above.

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
parent(Ax)           # parent array holding the local data (here, an Array{Float64,3})
size(Ax)             # total size of the array = (42, 31, 29)
size_local(Ax)       # size of local part

# Create another pencil configuration, decomposing along dimensions (1, 3).
# We could use the same constructor as before, but it's recommended to reuse the
# previous Pencil instead.
pen_y = Pencil(pen_x, decomp_dims=(1, 3))

# Now transpose from the x-pencil to the y-pencil configuration, redistributing
# the data initially in Ax.
Ay = PencilArray{Float64}(undef, pen_y)
transpose!(Ay, Ax)
```
