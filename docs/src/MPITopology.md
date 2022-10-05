# [MPI topology](@id sec:mpi_topology)

The [`MPITopology`](@ref) type defines the MPI Cartesian topology of the
decomposition.
In other words, it contains information about the number of decomposed
dimensions, and the number of processes in each of these dimensions.

This type should only be used if more control is needed regarding the MPI
decomposition.
In particular, dealing with `MPITopology` is not required when using the
[high-level interface](@ref pencil-high-level) to construct domain
decomposition configurations.

## Construction

The main `MPITopology` constructor takes a MPI communicator and a tuple
specifying the number of processes in each dimension.
For instance, to distribute 12 MPI processes on a $3 × 4$ grid:
```julia
comm = MPI.COMM_WORLD  # we assume MPI.Comm_size(comm) == 12
pdims = (3, 4)
topology = MPITopology(comm, pdims)
```

A convenience constructor is provided that automatically chooses a default
`pdims` from the number of processes and from the dimension `N` of
decomposition grid. For instance, for a two-dimensional decomposition:
```julia
topology = MPITopology(comm, Val(2))
```
Under the hood, this works by letting
[`MPI_Dims_create`](https://www.open-mpi.org/doc/current/man3/MPI_Dims_create.3.php)
choose the number of divisions along each dimension.

At the lower level, [`MPITopology`](@ref) uses
[`MPI_Cart_create`](https://www.open-mpi.org/doc/current/man3/MPI_Cart_create.3.php)
to define a Cartesian MPI communicator.
For more control, one can also create a Cartesian communicator using
[`MPI.Cart_create`](https://juliaparallel.org/MPI.jl/stable/reference/topology/#MPI.Cart_create),
and pass that to `MPITopology`:
```julia
dims = (3, 4)
comm_cart = MPI.Cart_create(comm, dims)
topology = MPITopology(comm_cart)
```

## Types

```@docs
MPITopology
```

## Methods

```@docs
get_comm(::MPITopology)
coords_local(::MPITopology)
length(::MPITopology)
ndims(::MPITopology)
size(::MPITopology)
```

## Index

```@index
Pages = ["MPITopology.md"]
```
