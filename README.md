# PencilArrays

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jipolanco.github.io/PencilArrays.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jipolanco.github.io/PencilArrays.jl/dev)

[![Build Status](https://travis-ci.com/jipolanco/PencilArrays.jl.svg?branch=master)](https://travis-ci.com/jipolanco/PencilArrays.jl)
[![Coverage](https://codecov.io/gh/jipolanco/PencilArrays.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jipolanco/PencilArrays.jl)

Distributed Julia arrays using the MPI protocol.

This package provides a convenient framework for working with multidimensional
Julia arrays distributed among MPI processes.

The name of this package originates from the decomposition of 3D domains along
two out of three dimensions, sometimes called *pencil* decomposition.
This is illustrated by the figure below, where each coloured block is managed
by a different MPI process.

<p align="center">
  <br/>
  <img width="85%" alt="Pencil decomposition of 3D domains" src="docs/src/img/pencils.svg">
</p>

Note that PencilArrays can decompose grids of arbitrary dimension `N`, along an
arbitrary number of subdimensions `M < N`.
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

- convenient parallel I/O using the Parallel HDF5 libraries;

- distributed FFTs and related transforms via the
  [PencilFFTs](https://github.com/jipolanco/PencilFFTs.jl) package.

## Installation

PencilArrays is in the process of being registered as a Julia package.
Then, it will be installable using the Julia package manager:

    julia> ] add PencilArrays

