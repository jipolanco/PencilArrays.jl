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

<p align="center">
  <br/>
  <img width="85%" alt="Pencil decomposition of 3D domains" src="docs/src/img/pencils.svg">
</p>

More generally, PencilArrays can decompose arrays of arbitrary dimension `N`,
along an arbitrary subset of `M` dimensions.
(In the example above, `N = 3` and `M = 2`.)

PencilArrays is the basis for the
[PencilFFTs](https://github.com/jipolanco/PencilFFTs.jl) package, which
provides efficient and highly scalable distributed FFTs.

## Features

- distribution of `N`-dimensional arrays among MPI processes;

- decomposition of arrays along an arbitrary subset of dimensions;

- transpositions between different decomposition configurations, using
  point-to-point and collective MPI communications;

- efficient and transparent arrays with permuted dimensions (similar to
  [`PermutedDimsArray`](https://docs.julialang.org/en/latest/base/arrays/#Base.PermutedDimsArrays.PermutedDimsArray));

- convenient parallel I/O using the [Parallel
  HDF5](https://portal.hdfgroup.org/display/HDF5/Parallel+HDF5) libraries;

- distributed FFTs and related transforms via the
  [PencilFFTs](https://github.com/jipolanco/PencilFFTs.jl) package.

## Installation

PencilArrays will soon be registered as a Julia package.
Then, it will be installable using the Julia package manager:

    julia> ] add PencilArrays

[^1]:
    Figure adapted from [this PhD thesis](https://hal.archives-ouvertes.fr/tel-02084215v1).
