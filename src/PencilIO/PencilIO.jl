module PencilIO

using ..PencilArrays
import ..PencilArrays: MaybePencilArrayCollection, collection_size

using MPI
using Requires: @require
using TimerOutputs

"""
    ParallelIODriver

Abstract type specifying a parallel I/O driver.
"""
abstract type ParallelIODriver end

"""
    open([f::Function], driver::ParallelIODriver, filename,
         comm::MPI.Comm, [info::MPI.Info]; keywords...)

Open parallel file using the chosen driver.

## I/O mode

Keyword arguments control the I/O mode.
Possible keywords are `read`, `write`, `create`, `append` and `truncate`,
although certain drivers ignore some of these keywords (see below).

Mode keywords have the same behaviour and defaults as `Base.open`.

## Parallel drivers

The following drivers are supported:

- [`PHDF5Driver`](@ref)
- [`MPIIODriver`](@ref): the `truncate` keyword is ignored.
"""
function Base.open(f::Function, driver::ParallelIODriver, args...; kw...)
    fid = open(driver, args...; kw...)
    try
        f(fid)
    finally
        close(fid)
    end
end

include("mpi_io.jl")

function __init__()
    @require HDF5="f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f" @eval include("hdf5.jl")
end

end
