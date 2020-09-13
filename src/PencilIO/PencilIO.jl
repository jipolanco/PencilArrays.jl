module PencilIO

using ..PencilArrays
import ..PencilArrays: MaybePencilArrayCollection, collection_size

using MPI
using Requires: @require
using StaticArrays: SVector
using TimerOutputs

"""
    ParallelIODriver

Abstract type specifying a parallel I/O driver.
"""
abstract type ParallelIODriver end

"""
    open([f::Function], driver::ParallelIODriver, filename, comm::MPI.Comm; keywords...)

Open parallel file using the chosen driver.

## Keyword arguments

Supported keyword arguments include:

- open mode arguments: `read`, `write`, `create`, `append` and `truncate`.
  These have the same behaviour and defaults as `Base.open`.
  Some of them may not be supported by the chosen driver (see below).

- as in [`MPI.File.open`](https://juliaparallel.github.io/MPI.jl/latest/io/#MPI.File.open),
  other arguments are passed via an `MPI.Info` object.

Note that driver-specific options (such as HDF5 property lists) must be passed
to each driver's constructor.

## Parallel I/O drivers

The following drivers are supported:

- [`PHDF5Driver`](@ref)

- [`MPIIODriver`](@ref)
    * setting `append = true` is unsupported and throws `ArgumentError`
    * the `truncate` keyword is ignored
"""
function Base.open(f::Function, driver::ParallelIODriver, args...; kw...)
    fid = open(driver, args...; kw...)
    try
        f(fid)
    finally
        close(fid)
    end
end

function metadata(x::MaybePencilArrayCollection)
    pen = pencil(x)
    topo = topology(x)
    (
        permutation = Tuple(get_permutation(x)),
        extra_dims = extra_dims(x),
        decomposed_dims = get_decomposition(pen),
        process_dims = size(topo),
    )
end

include("mpi_io.jl")

function __init__()
    @require HDF5="f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f" @eval include("hdf5.jl")
end

end
