# TODO
# - provide setindex! for compat with HDF5
# - support array collections?

export MPIIODriver

"""
    MPIIODriver(; sequential=false, uniqueopen=false, deleteonclose=false)

MPI-IO driver using the MPI.jl package.

Keyword arguments are passed to
[`MPI.File.open`](https://juliaparallel.github.io/MPI.jl/latest/io/#MPI.File.open).
"""
Base.@kwdef struct MPIIODriver <: ParallelIODriver
    sequential    :: Bool = false
    uniqueopen    :: Bool = false
    deleteonclose :: Bool = false
end

Base.open(D::MPIIODriver, filename::AbstractString, comm::MPI.Comm; keywords...) =
    MPI.File.open(
        comm, filename;
        sequential=D.sequential, uniqueopen=D.uniqueopen,
        deleteonclose=D.deleteonclose, keywords...,
    )

"""
    write(ff::MPI.FileHandle, x::PencilArray;
          chunks=false, collective=true, infokws...)

Write [`PencilArray`](@ref) to binary file using MPI-IO.

# Optional arguments

- if `chunks = true`, data is written in contiguous blocks, with one block per
  process.
  Otherwise, each process writes to discontiguous sections of disk, using
  `MPI.File.set_view!` and custom datatypes.
  Note that discontiguous I/O (the default) is more convenient, as it allows to
  read back the data using a different number or distribution of MPI processes.

- if `collective = true`, the dataset is written collectivelly. This is
  usually recommended for performance.

- when writing discontiguous blocks, additional keyword arguments are passed via
  a `MPI.Info` object to `MPI.File.set_view!`. This is ignored if `chunks = true`.

"""
function Base.write(ff::MPI.FileHandle, x::PencilArray;
                    collective=true, chunks=false, kw...)
    if chunks
        write_contiguous(ff, x; collective=collective, kw...)
    else
        write_discontiguous(ff, x; collective=collective, kw...)
    end
    nothing
end

function write_discontiguous(ff::MPI.FileHandle, x::PencilArray;
                             collective, infokws...)
    to = get_timer(pencil(x))
    @timeit_debug to "Write MPI-IO discontiguous" begin
        disp = 0
        etype = MPI.Datatype(eltype(x))
        filetype = create_discontiguous_datatype(x, MemoryOrder())  # TODO cache datatype?
        MPI.File.set_view!(ff, 0, etype, filetype; infokws...)
        A = parent(x)
        if collective
            MPI_File_write_all(ff, A)
        else
            MPI_File_write(ff, A)
        end
    end
    nothing
end

# TODO add these to MPI.jl
function MPI_File_write(file::MPI.FileHandle, buf::MPI.Buffer)
    stat_ref = Ref{MPI.Status}(MPI.STATUS_EMPTY)
    # int MPI_File_write(MPI_File fh, const void *buf,
    #                    int count, MPI_Datatype datatype, MPI_Status *status)
    MPI.@mpichk ccall((:MPI_File_write, MPI.libmpi), Cint,
                      (MPI.MPI_File, MPI.MPIPtr, Cint, MPI.MPI_Datatype, Ptr{MPI.Status}),
                      file, buf.data, buf.count, buf.datatype, stat_ref)
    return stat_ref[]
end

function MPI_File_write_all(file::MPI.FileHandle, buf::MPI.Buffer)
    stat_ref = Ref{MPI.Status}(MPI.STATUS_EMPTY)
    # int MPI_File_write_all(MPI_File fh, const void *buf,
    #                        int count, MPI_Datatype datatype, MPI_Status *status)
    MPI.@mpichk ccall((:MPI_File_write_all, MPI.libmpi), Cint,
                      (MPI.MPI_File, MPI.MPIPtr, Cint, MPI.MPI_Datatype, Ptr{MPI.Status}),
                      file, buf.data, buf.count, buf.datatype, stat_ref)
    return stat_ref[]
end

MPI_File_write(file::MPI.FileHandle, data) = MPI_File_write(file, MPI.Buffer_send(data))
MPI_File_write_all(file::MPI.FileHandle, data) = MPI_File_write_all(file, MPI.Buffer_send(data))

function create_discontiguous_datatype(x::PencilArray, order=MemoryOrder())
    sizes = size_global(x, order)
    subsizes = size_local(x, order)
    offsets = map(r -> first(r) - 1, range_local(x, order))
    oldtype = MPI.Datatype(eltype(x), commit=false)
    dtype = MPI.Types.create_subarray(sizes, subsizes, offsets, oldtype)
    MPI.Types.commit!(dtype)
    dtype
end

function write_contiguous(ff::MPI.FileHandle, x::PencilArray;
                          collective, infokws...)
    to = get_timer(pencil(x))
    @timeit_debug to "Write MPI-IO contiguous" begin
        offset = mpi_io_offset(x)
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
