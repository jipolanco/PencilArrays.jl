module PencilIO

using ..PencilArrays
import ..PencilArrays: MaybePencilArrayCollection, collection_size, collection

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
  Some of them may be ignored by the chosen driver (see driver-specific docs).

- as in [`MPI.File.open`](https://juliaparallel.github.io/MPI.jl/latest/io/#MPI.File.open),
  other arguments are passed via an `MPI.Info` object.

Note that driver-specific options (such as HDF5 property lists) must be passed
to each driver's constructor.

## See also

- [`open(::MPIIODriver)`](@ref) for MPI-IO specific options
- [`open(::PHDF5Driver)`](@ref) for HDF5 specific options

"""
function Base.open(::ParallelIODriver) end

function Base.open(f::Function, driver::ParallelIODriver, args...; kw...)
    fid = open(driver, args...; kw...)
    try
        f(fid)
    finally
        close(fid)
    end
end

# Metadata to be attached to each dataset (as HDF5 attributes or in an external
# metadata file).
function metadata(x::MaybePencilArrayCollection)
    pen = pencil(x)
    topo = topology(x)
    edims = extra_dims(x)  # this may be an empty tuple, with no type information
    (
        permutation = Tuple(get_permutation(x)),
        extra_dims = SVector{length(edims),Int}(edims),
        decomposed_dims = get_decomposition(pen),
        process_dims = size(topo),
    )
end

function keywords_to_open(; read=nothing, write=nothing, create=nothing,
                          truncate=nothing, append=nothing, other_kws...)
    flags = Base.open_flags(read=read, write=write, create=create,
                            truncate=truncate, append=append)
    flags, other_kws
end

include("mpi_io.jl")

function __init__()
    @require HDF5="f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f" @eval include("hdf5.jl")
end

end
