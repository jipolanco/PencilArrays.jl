# Reductions

Reduction over [`PencilArray`](@ref)s using Julia functions such as `minimum`,
`maximum` or `sum` are performed on the global data.
This involves first a local reduction over each process, followed by a global
reduction of a scalar quantity using
[`MPI.Allreduce`](https://juliaparallel.github.io/MPI.jl/latest/collective/#MPI.Allreduce).

For example:

```julia
using MPI
using PencilArrays

MPI.Init()

comm = MPI.COMM_WORLD
nprocs = MPI.Comm_size(comm)
rank = MPI.Comm_rank(comm)
id = rank + 1

pen = Pencil((16, 32, 14), comm)
u = PencilArray{Int}(undef, pen)
fill!(u, 2 * id)

minimum(u)  # = 2
maximum(u)  # = 2 * nprocs

minimum(abs2, u)  # = 4
maximum(abs2, u)  # = (2 * nprocs)^2

```

!!! note "Note on associativity"

    Associative reduction operations like
    [`foldl`](https://docs.julialang.org/en/v1/base/collections/#Base.foldl-Tuple{Any,%20Any}),
    [`foldr`](https://docs.julialang.org/en/v1/base/collections/#Base.foldr-Tuple{Any,%20Any}),
    [`mapfoldl`](https://docs.julialang.org/en/v1/base/collections/#Base.mapfoldl-Tuple{Any,%20Any,%20Any})
    and
    [`mapfoldr`](https://docs.julialang.org/en/v1/base/collections/#Base.mapfoldr-Tuple{Any,%20Any,%20Any})
    are also defined for consistency, but these operations are not
    guaranteed to strictly respect left or right associativity.
    In fact, associativity is only respected on each local process, before
    results are reduced among all processes.
