export MPIIODriver

"""
    MPIIODriver(; sequential=false, uniqueopen=false, deleteonclose=false)

MPI-IO driver using the MPI.jl package.

Keyword arguments are passed to
[`MPI.File.open`](https://juliaparallel.github.io/MPI.jl/latest/io/#MPI.File.open).
"""
struct MPIIODriver <: ParallelIODriver
    sequential :: Bool
    uniqueopen :: Bool
    deleteonclose :: Bool
    MPIIODriver(; sequential=false, uniqueopen=false, deleteonclose=false) =
        new(sequential, uniqueopen, deleteonclose)
end

Base.open(D::MPIIODriver, filename::AbstractString, comm::MPI.Comm; keywords...) =
    MPI.File.open(
        comm, filename;
        sequential=D.sequential, uniqueopen=D.uniqueopen,
        deleteonclose=D.deleteonclose, keywords...,
    )

# TODO
# - provide setindex! for compat with HDF5
# - collective / independent I/O
# - support array collections?
# - write data into one block per rank
# - optionally write metadata file?

"""
    write(ff::MPI.FileHandle, x::PencilArray; collective=true)

Write [`PencilArray`](@ref) to binary file using MPI-IO.

Each process writes to a separate block. Blocks are sorted by MPI rank.
Data must be read back with the same number and distribution of MPI processes
used for writing.
"""
function Base.write(ff::MPI.FileHandle, x::PencilArray; collective=true)
    to = get_timer(pencil(x))
    @timeit_debug to "Write MPI-IO" begin
        offset = mpi_io_offset(x)
        # dtype = MPI.Datatype(T)
        # MPI.File.set_view!(ff, dest, dtype, dtype)
        A = parent(x)
        if collective
            MPI.File.write_at_all(ff, offset, A)
        else
            MPI.File.write_at(ff, offset, A)
        end
    end
    nothing
end

function mpi_io_offset(x::PencilArray)
    topo = topology(pencil(x))
    # Linear index of this process in the topology.
    # (TODO This should be stored in MPITopology...)
    n = LinearIndices(topo)[coords_local(topo)...]
    off = 0
    for m = 1:(n - 1)
        r = range_remote(x, m)
        off += length(CartesianIndices(r))  # length of data held by remote process
    end
    T = eltype(x)
    off * sizeof(T)
end
