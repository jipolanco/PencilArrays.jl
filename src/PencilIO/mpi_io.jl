export MPIIODriver

import JSON3
import JSON3.StructTypes

# Version of internal MPIIO format.
# If the version is updated, it should match the upcoming PencilArrays version.
const MPIIO_VERSION = v"0.3"

StructTypes.StructType(::Type{typeof(MPIIO_VERSION)}) = StructTypes.Struct()

"""
    MPIIODriver(; sequential = false, uniqueopen = false, deleteonclose = false)

MPI-IO driver using the MPI.jl package.

Keyword arguments are passed to
[`MPI.File.open`](https://juliaparallel.github.io/MPI.jl/latest/io/#MPI.File.open).
"""
Base.@kwdef struct MPIIODriver <: ParallelIODriver
    sequential    :: Bool = false
    uniqueopen    :: Bool = false
    deleteonclose :: Bool = false
end

const MetadataDict = Dict{Symbol,Any}
const DatasetKey = Symbol
const DatasetDict = Dict{DatasetKey,Any}

"""
    MPIFile

Wraps a `MPI.FileHandle`, also including file position information and metadata.

File position is updated when reading and writing data, and is independent of
the individual and shared file pointers defined by MPI.
"""
mutable struct MPIFile
    file       :: MPI.FileHandle
    comm       :: MPI.Comm
    filename   :: String
    meta       :: MetadataDict
    position   :: Int  # file position in bytes
    write_mode :: Bool
    MPIFile(file, comm, filename, meta; write) =
        new(file, comm, filename, meta, 0, write)
end

function MPIFile(comm::MPI.Comm, filename; kws...)
    flags, other_kws = keywords_to_open(; kws...)
    meta = if flags.write && !flags.append
        mpiio_init_metadata()
    else
        mpiio_load_metadata(filename_meta(filename))
    end
    file = MPIFile(
        MPI.File.open(comm, filename; kws...),
        comm, filename, meta, write=flags.write,
    )
    if flags.append
        # Synchronise position in file.
        pos = MPI.File.get_position_shared(parent(file))
        seek(file, pos)
    end
    file
end

mpiio_init_metadata() = MetadataDict(
    :driver => (type = "MPIIODriver", version = MPIIO_VERSION),
    :datasets => DatasetDict(),
)

function mpiio_load_metadata(filename)
    isfile(filename) || error("metadata file not found: $filename")
    meta = open(JSON3.read, filename, "r")
    # Convert from specific JSON3 type to Dict, so that datasets can be appended.
    MetadataDict(
        :driver => meta.driver,
        :datasets => DatasetDict(meta.datasets),
    )
end

function Base.close(ff::MPIFile)
    if should_write_metadata(ff)
        write_metadata(ff, metadata(ff))
    end
    close(parent(ff))
end

function write_metadata(ff, meta)
    MPI.Comm_rank(ff.comm) == 0 || return
    open(filename_meta(ff), "w") do io
        JSON3.write(io, meta)
        write(io, '\n')
    end
    nothing
end

filename_meta(fname) = string(fname, ".json")
filename_meta(ff::MPIFile) = filename_meta(get_filename(ff))
metadata(ff::MPIFile) = ff.meta
should_write_metadata(ff::MPIFile) = ff.write_mode
get_filename(ff::MPIFile) = ff.filename
Base.parent(ff::MPIFile) = ff.file
Base.position(ff::MPIFile) = ff.position
Base.skip(ff::MPIFile, offset) = ff.position += offset
Base.seek(ff::MPIFile, pos) = ff.position = pos

Base.open(D::MPIIODriver, filename::AbstractString, comm::MPI.Comm; keywords...) =
    MPIFile(
        comm, filename;
        sequential=D.sequential, uniqueopen=D.uniqueopen,
        deleteonclose=D.deleteonclose, keywords...,
    )

"""
    setindex!(file::MPIFile, x::PencilArray, name::AbstractString;
              chunks = false, collective = true, infokws...)

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
  an `MPI.Info` object to `MPI.File.set_view!`. This is ignored if `chunks = true`.

"""
function Base.setindex!(ff::MPIFile, x::PencilArray, name::AbstractString;
                        collective=true, chunks=false, kw...)
    file = parent(ff)
    offset = position(ff)
    if chunks
        write_contiguous(file, x; offset=offset, collective=collective, kw...)
    else
        write_discontiguous(file, x; offset=offset, collective=collective, kw...)
    end
    add_metadata(ff, x, name, chunks)
    skip(ff, sizeof_global(x))
    x
end

function add_metadata(file::MPIFile, x, name, chunks::Bool)
    meta = metadata(file)
    meta === nothing && return
    meta[:datasets][DatasetKey(name)] = (
        metadata(x)...,
        element_type = eltype(x),
        dims_logical = size_global(x, LogicalOrder()),
        dims_memory = size_global(x, MemoryOrder()),
        chunks = chunks,
        offset_bytes = position(file),
        size_bytes = sizeof_global(x),
    )
    nothing
end

"""
    read!(file::MPIFile, x::PencilArray, name::AbstractString;
          collective = true, infokws...)

Read binary data from an MPI-IO stream, filling in [`PencilArray`](@ref).

See [`setindex!`](@ref setindex!(::MPIFile)) for details on keyword arguments.
"""
function Base.read!(ff::MPIFile, x::PencilArray, name::AbstractString;
                    collective=true, kw...)
    meta = get(metadata(ff)[:datasets], DatasetKey(name), nothing)
    meta === nothing && error("dataset '$name' not found")
    file = parent(ff)
    offset = meta.offset_bytes :: Int
    chunks = meta.chunks :: Bool
    check_metadata(x, meta.element_type, Tuple(meta.dims_memory), meta.size_bytes)
    if chunks
        check_read_chunks(x, meta.process_dims, name)
        read_contiguous!(file, x; offset=offset, collective=collective, kw...)
    else
        read_discontiguous!(file, x; offset=offset, collective=collective, kw...)
    end
    x
end

function check_metadata(x, file_eltype, file_dims, file_sizeof)
    T = eltype(x)
    if string(T) != file_eltype
        error("incompatible type of file and array: $file_eltype ≠ $T")
    end
    sz = (size_global(x, MemoryOrder())..., collection_size(x)...)
    if sz !== file_dims
        error("incompatible dimensions of dataset in file and array: $file_dims ≠ $sz")
    end
    @assert sizeof_global(x) == file_sizeof
    nothing
end

function check_read_chunks(x, pdims_file, name)
    pdims = size(topology(x))
    if length(pdims) != length(pdims_file) || any(pdims .!= pdims_file)
        error("dataset '$name' was written in chunks with a different MPI topology" *
              " ($pdims ≠ $pdims_file)")
    end
    nothing
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
