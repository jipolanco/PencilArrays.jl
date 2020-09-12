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

"""
    MPIFile

Wraps a `MPI.FileHandle`, also including file position information.

File position is updated when reading and writing data, and is independent of
the individual and shared file pointers defined by MPI.
"""
mutable struct MPIFile
    file     :: MPI.FileHandle
    position :: Int  # file position in bytes
    MPIFile(file) = new(file, 0)
end

function MPIFile(comm::MPI.Comm, filename; append=false, kw...)
    # Append mode is not supported.
    # Appending can be unpredictable, because the MPI implementation may pad
    # each dataset (or file?) with extra (mega)bytes, I guess for alignment
    # purposes.
    if append
        throw(ArgumentError("append=true not supported by MPI-IO driver"))
    end
    MPIFile(MPI.File.open(comm, filename; append=append, kw...))
end

Base.parent(ff::MPIFile) = ff.file
Base.close(ff::MPIFile) = close(parent(ff))
Base.position(ff::MPIFile) = ff.position
Base.skip(ff::MPIFile, offset) = ff.position += offset

Base.open(D::MPIIODriver, filename::AbstractString, comm::MPI.Comm; keywords...) =
    MPIFile(
        comm, filename;
        sequential=D.sequential, uniqueopen=D.uniqueopen,
        deleteonclose=D.deleteonclose, keywords...,
    )

"""
    write(file::MPIFile, x::PencilArray;
          chunks = false, collective = true, infokws...)

Write [`PencilArray`](@ref) to binary file using MPI-IO.

Returns the number of bytes written by all processes.

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
  an `MPI.Info` object to `MPI.File.set_view!`. This is ignored if `chunks = true`.

"""
function Base.write(ff::MPIFile, x::PencilArray;
                    collective=true, chunks=false, kw...)
    file = parent(ff)
    offset = position(ff)
    if chunks
        write_contiguous(file, x; offset=offset, collective=collective, kw...)
    else
        write_discontiguous(file, x; offset=offset, collective=collective, kw...)
    end
    nb = sizeof_global(x)
    skip(ff, nb)
    nb
end

"""
    read!(file::MPIFile, x::PencilArray;
          chunks = false, collective = true, infokws...)

Read binary data from an MPI-IO stream, filling in [`PencilArray`](@ref).

See [`write`](@ref write(::MPIFile)) for details on keyword arguments.
"""
function Base.read!(ff::MPIFile, x::PencilArray;
                    collective=true, chunks=false, kw...)
    file = parent(ff)
    offset = position(ff)
    if chunks
        read_contiguous!(file, x; offset=offset, collective=collective, kw...)
    else
        read_discontiguous!(file, x; offset=offset, collective=collective, kw...)
    end
    skip(ff, sizeof_global(x))
    x
end

function write_discontiguous(ff::MPI.FileHandle, x::PencilArray;
                             offset, collective, infokws...)
    to = get_timer(pencil(x))
    @timeit_debug to "Write MPI-IO discontiguous" begin
        set_view!(ff, x, offset; infokws...)
        A = parent(x)
        if collective
            MPI_File_write_all(ff, A)
        else
            MPI_File_write(ff, A)
        end
    end
    nothing
end

function read_discontiguous!(ff::MPI.FileHandle, x::PencilArray;
                             offset, collective, infokws...)
    to = get_timer(pencil(x))
    @timeit_debug to "Read MPI-IO discontiguous" begin
        set_view!(ff, x, offset; infokws...)
        A = parent(x)
        if collective
            MPI_File_read_all!(ff, A)
        else
            MPI_File_read!(ff, A)
        end
    end
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

function MPI_File_read!(file::MPI.FileHandle, buf::MPI.Buffer)
    stat_ref = Ref{MPI.Status}(MPI.STATUS_EMPTY)
    # int MPI_File_read(MPI_File fh, void *buf,
    #                   int count, MPI_Datatype datatype, MPI_Status *status)
    MPI.@mpichk ccall((:MPI_File_read, MPI.libmpi), Cint,
                      (MPI.MPI_File, MPI.MPIPtr, Cint, MPI.MPI_Datatype, Ptr{MPI.Status}),
                      file, buf.data, buf.count, buf.datatype, stat_ref)
    return stat_ref[]
end

function MPI_File_read_all!(file::MPI.FileHandle, buf::MPI.Buffer)
    stat_ref = Ref{MPI.Status}(MPI.STATUS_EMPTY)
    # int MPI_File_read_all(MPI_File fh, void *buf,
    #                       int count, MPI_Datatype datatype, MPI_Status *status)
    MPI.@mpichk ccall((:MPI_File_read_all, MPI.libmpi), Cint,
                      (MPI.MPI_File, MPI.MPIPtr, Cint, MPI.MPI_Datatype, Ptr{MPI.Status}),
                      file, buf.data, buf.count, buf.datatype, stat_ref)
    return stat_ref[]
end

for f in (:MPI_File_write, :MPI_File_write_all)
    @eval $f(file::MPI.FileHandle, data) = $f(file, MPI.Buffer_send(data))
end
for f in (:MPI_File_read!, :MPI_File_read_all!)
    @eval $f(file::MPI.FileHandle, data) = $f(file, MPI.Buffer(data))
end

function set_view!(ff, x::PencilArray, offset; infokws...)
    etype = MPI.Datatype(eltype(x))
    filetype = create_discontiguous_datatype(x, MemoryOrder())  # TODO cache datatype?
    MPI.File.set_view!(ff, offset, etype, filetype; infokws...)
    nothing
end

function create_discontiguous_datatype(x::PencilArray, order)
    sizes = size_global(x, order)
    subsizes = size_local(x, order)
    offsets = map(r -> first(r) - 1, range_local(x, order))
    oldtype = MPI.Datatype(eltype(x), commit=false)
    dtype = MPI.Types.create_subarray(sizes, subsizes, offsets, oldtype)
    MPI.Types.commit!(dtype)
    dtype
end

function write_contiguous(ff::MPI.FileHandle, x::PencilArray;
                          offset, collective, infokws...)
    to = get_timer(pencil(x))
    @timeit_debug to "Write MPI-IO contiguous" begin
        offset += mpi_io_offset(x)
        A = parent(x)
        if collective
            MPI.File.write_at_all(ff, offset, A)
        else
            MPI.File.write_at(ff, offset, A)
        end
    end
    nothing
end

function read_contiguous!(ff::MPI.FileHandle, x::PencilArray;
                          offset, collective, infokws...)
    to = get_timer(pencil(x))
    @timeit_debug to "Read MPI-IO contiguous" begin
        offset += mpi_io_offset(x)
        A = parent(x)
        if collective
            MPI.File.read_at_all!(ff, offset, A)
        else
            MPI.File.read_at!(ff, offset, A)
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
