```@meta
CurrentModule = PencilArrays.Pencils
```

# [Pencil configurations](@id sec:pencil_configs)

A *pencil* configuration refers to a given distribution of multidimensional
data among MPI processes.
This information is encoded in the [`Pencil`](@ref) type.

A pencil configuration includes:
- [MPI topology](@ref sec:mpi_topology) information,
- global and local dimensions of the numerical grid,
- subset of decomposed dimensions,
- definition of optional permutation of dimensions.

## Construction

The creation of a new [`Pencil`](@ref) requires a [`MPITopology`](@ref) and
the global data dimensions.
One may also specify the list of decomposed dimensions, as well as an optional
permutation of dimensions.

For instance, to decompose along 2 dimensions of a 3D dataset,

```julia
topology = MPITopology(comm, (8, 4))  # assuming 8×4 = 32 processes
dims_global = (16, 32, 64)
pencil = Pencil(topology, dims_global)
```

By default, the decomposed dimensions are the rightmost ones (in this case,
dimensions `2` and `3`). A different set of dimensions may be selected via the
optional positional argument. For instance, to decompose along dimensions `1`
and `3` instead,

```julia
decomp_dims = (1, 3)
pencil = Pencil(topology, dims_global, decomp_dims)
```

One may also want to work with multiple pencil configurations that differ, for
instance, on the selection of decomposed dimensions.
For this case, a second constructor is available that takes an already existing
`Pencil` instance.
Calling this constructor should be preferred when possible since it allows
sharing memory buffers (used for instance for [global transpositions](@ref
Global-MPI-operations)) and thus reducing memory usage.
The following creates a `Pencil` equivalent to the one above, but with
different decomposed dimensions:
```julia
pencil_x = Pencil(pencil; decomp_dims=(1, 2))
```
See the [`Pencil`](@ref) documentation for more details.

## Dimension permutations

As mentioned above, a `Pencil` may optionally be given information on dimension
permutations.
In this case, the layout of the data arrays in memory is different from the
logical order of dimensions.
For performance reasons, permutations are compile-time objects defined in the
[StaticPermutations](https://github.com/jipolanco/StaticPermutations.jl)
package.

To make permutations clearer, consider the example above where the global data
dimensions are $N_x × N_y × N_z = 16 × 32 × 64$.
In this case, the logical order is $(x, y, z)$.
Now let's say that we want the memory order of the data to be $(y, z, x)$,[^1]
which corresponds to the permutation `(2, 3, 1)`.

Permutations are passed to the `Pencil` constructor via the `permute` keyword
argument.
Dimension permutations should be specified using a
[`Permutation`](https://jipolanco.github.io/StaticPermutations.jl/stable/#StaticPermutations.Permutation)
object.
For instance,
```julia
permutation = Permutation(2, 3, 1)
pencil = Pencil(#= ... =#, permute=permutation)
```
One can also pass a
[`NoPermutation`](https://jipolanco.github.io/StaticPermutations.jl/stable/#StaticPermutations.NoPermutation)
to disable permutations (this is the default).

## Types

```@docs
Pencil
Pencils.AbstractIndexOrder
MemoryOrder
LogicalOrder
```

## Methods

```@docs
topology(::Pencil)
get_comm(::Pencil)
decomposition(::Pencil)
permutation(::Pencil)
timer(::Pencil)
length(::Pencil)
ndims(::Pencil)
range_remote(::Pencil, ::Integer, ::LogicalOrder)
range_local(::Pencil, ::LogicalOrder)
size_global(::Pencil, ::LogicalOrder)
size_local(::Pencil, etc...)
to_local(::Pencil)
```

## Index

```@index
Pages = ["Pencils.md"]
```

[^1]:
    Why would we want this?
    One application is to efficiently perform FFTs along $y$, which, under
    this permutation, would be the fastest dimension.
    This is used by the [PencilFFTs](https://github.com/jipolanco/PencilFFTs.jl) package.
