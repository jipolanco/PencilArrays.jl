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

### High-level interface

The simplest way of constructing a new [`Pencil`](@ref) is by passing the
global dataset dimensions and an MPI communicator to the `Pencil` constructor:

```julia-repl
julia> dims_global = (16, 32, 64);

julia> comm = MPI.COMM_WORLD;

julia> pen = Pencil(dims_global, comm)
Decomposition of 3D data
    Data dimensions: (16, 32, 64)
    Decomposed dimensions: (2, 3)
    Data permutation: NoPermutation()
```

This will decompose the data along the two last dimensions of the dataset:[^1]

```julia-repl
julia> decomposition(pen)
(2, 3)
```

For instance, if the communicator `comm` has 4 MPI processes, then each process will hold a subset of data of size `(16, 16, 32)`:

```julia-repl
julia> topology(pen)
MPI topology: 2D decomposition (2×2 processes)  # assuming MPI.Comm_size(comm) == 4

julia> size_local(pen)
(16, 16, 32)
```

Instead of the default, one may want to choose the subset of dimensions that
should be decomposed.
For instance, to decompose along the first dimension only:

```julia-repl
julia> decomp_dims = (1,);

julia> pen = Pencil(dims_global, decomp_dims, comm)
Decomposition of 3D data
    Data dimensions: (16, 32, 64)
    Decomposed dimensions: (1,)
    Data permutation: NoPermutation()

julia> decomposition(pen)
(1,)

julia> topology(pen)
MPI topology: 1D decomposition (4 processes)  # again, assuming 4 MPI processes
```


### Low-level interface

Note that the above high-level constructors don't require the definition of
a [`MPITopology`](@ref), which is constructed implicitly.
For more control, one may want to manually construct a `MPITopology`, and then construct a `Pencil` from that.
As above, one may also specify the list of decomposed dimensions.

For instance, we may want to decompose 32 MPI processes into a 8×4 Cartesian
topology.
This is done as follows:

```julia-repl
julia> topo = MPITopology(comm, (8, 4))  # NOTE: fails if MPI.Comm_size(comm) ≠ 32
MPI topology: 2D decomposition (8×4 processes)

julia> dims_global = (16, 32, 64);

julia> pen = Pencil(topo, dims_global)
Decomposition of 3D data
    Data dimensions: (16, 32, 64)
    Decomposed dimensions: (2, 3)
    Data permutation: NoPermutation()
```

As before, the decomposed dimensions are the rightmost ones by default (in this
case, dimensions `2` and `3`). A different set of dimensions may be
selected via an optional positional argument.
For instance, to decompose along dimensions `1` and `3` instead:

```julia-repl
julia> decomp_dims = (1, 3);

julia> pen = Pencil(topo, dims_global, decomp_dims)
Decomposition of 3D data
    Data dimensions: (16, 32, 64)
    Decomposed dimensions: (1, 3)
    Data permutation: NoPermutation()
```

### Defining multiple pencils

One may also want to work with multiple pencil configurations that differ, for
instance, on the selection of decomposed dimensions.
For this case, a constructor is available that takes an already existing
`Pencil` instance.
Calling this constructor should be preferred when possible since it allows
sharing memory buffers (used for instance for [global transpositions](@ref
Global-MPI-operations)) and thus reducing memory usage.
The following creates a `Pencil` equivalent to the one above, but with
different decomposed dimensions:

```julia-repl
julia> pen_y = Pencil(pen; decomp_dims = (1, 3))
Decomposition of 3D data
    Data dimensions: (16, 32, 64)
    Decomposed dimensions: (1, 3)
    Data permutation: NoPermutation()
```

See the [`Pencil`](@ref) documentation for more details.

## Dimension permutations

A `Pencil` may optionally be given information on dimension permutations.
In this case, the layout of the data arrays in memory is different from the
logical order of dimensions.
For performance reasons, permutations are compile-time objects defined in the
[StaticPermutations](https://github.com/jipolanco/StaticPermutations.jl)
package.

To make permutations clearer, consider the example above where the global data
dimensions are $N_x × N_y × N_z = 16 × 32 × 64$.
In this case, the logical order is $(x, y, z)$.
Now let's say that we want the memory order of the data to be $(y, z, x)$,[^2]
which corresponds to the permutation `(2, 3, 1)`.

Permutations are passed to the `Pencil` constructor via the `permute` keyword
argument.
Dimension permutations should be specified using a
[`Permutation`](https://jipolanco.github.io/StaticPermutations.jl/stable/#StaticPermutations.Permutation)
object.
For instance,

```julia-repl
julia> perm = Permutation(2, 3, 1);

julia> pen = Pencil(dims_global, comm; permute = perm)
Decomposition of 3D data
    Data dimensions: (16, 32, 64)
    Decomposed dimensions: (2, 3)
    Data permutation: Permutation(2, 3, 1)
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
size(::Pencil)
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
    More generally, an ``N``-dimensional dataset is by default decomposed along its ``N - 1`` last dimensions.

[^2]:
    Why would we want this?
    One application is to efficiently perform FFTs along $y$, which, under
    this permutation, would be the fastest dimension.
    This is used by the [PencilFFTs](https://github.com/jipolanco/PencilFFTs.jl) package.
