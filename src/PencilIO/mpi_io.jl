export MPIIODriver

import JSON3, VersionParsing

# Version of internal MPIIO format.
# If the version is updated, it should match the upcoming PencilArrays version.
const MPIIO_VERSION = v"0.9.4"

const IS_LITTLE_ENDIAN = ENDIAN_BOM == 0x04030201

"""
    MPIIODriver(; sequential = false, uniqueopen = false, deleteonclose = false)

MPI-IO driver using the MPI.jl package.

Keyword arguments are passed to
[`MPI.File.open`](https://juliaparallel.github.io/MPI.jl/latest/io/#MPI.File.open).

This driver writes binary data along with a JSON file containing metadata.
When reading data, this JSON file is expected to be present along with the raw
data file.
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
    :driver => (type = "MPIIODriver", version = string(MPIIO_VERSION)),
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
        JSON3.pretty(io, JSON3.write(meta))
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

mpiio_version(ff::MPIFile) = mpiio_version(metadata(ff))

mpiio_version(meta::MetadataDict) =
    VersionParsing.vparse(string(meta[:driver][:version]))

"""
    open([f::Function], driver::MPIIODriver, filename, comm::MPI.Comm; keywords...)

Open parallel file using the MPI-IO driver.

See [`open(::ParallelIODriver)`](@ref) for common options for all drivers.

Driver-specific options may be passed via the `driver` argument. See
[`MPIIODriver`](@ref) for details.

## Driver notes

- the `truncate` keyword is ignored.
"""
function Base.open(::MPIIODriver) end

Base.open(D::MPIIODriver, filename::AbstractString, comm::MPI.Comm; keywords...) =
    MPIFile(
        comm, filename;
        sequential=D.sequential, uniqueopen=D.uniqueopen,
        deleteonclose=D.deleteonclose, keywords...,
    )

"""
    setindex!(file::MPIFile, x::MaybePencilArrayCollection,
              name::AbstractString; chunks = false, collective = true, infokws...)

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
function Base.setindex!(
        ff::MPIFile, x::MaybePencilArrayCollection, name::AbstractString;
        collective=true, chunks=false, kw...,
    )
    file = parent(ff)
    offset = position(ff)
    for u in collection(x)
        # TODO write all collection components at once (should be easier in the
        # discontiguous case)
        if chunks
            write_contiguous(file, u; offset=offset, collective=collective, kw...)
        else
            write_discontiguous(file, u; offset=offset, collective=collective, kw...)
        end
        offset += sizeof_global(u)
    end
    add_metadata(ff, x, name, chunks)
    skip(ff, sizeof_global(x))
    x
end

eltype_collection(x::PencilArray) = eltype(x)
eltype_collection(x::PencilArrayCollection) = eltype(first(x))

function add_metadata(file::MPIFile, x, name, chunks::Bool)
    meta = metadata(file)
    size_col = collection_size(x)
    size_log = size_global(x, LogicalOrder())
    size_mem = size_global(x, MemoryOrder())
    meta[:datasets][DatasetKey(name)] = (
        metadata(x)...,
        julia_endian_bom = repr(ENDIAN_BOM),  # write it as a string such as 0x04030201
        little_endian = IS_LITTLE_ENDIAN,
        element_type = eltype_collection(x),
        dims_logical = (size_log..., size_col...),
        dims_memory = (size_mem..., size_col...),
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
function Base.read!(ff::MPIFile, x::MaybePencilArrayCollection, name::AbstractString;
                    collective=true, kw...)
    meta = get(metadata(ff)[:datasets], DatasetKey(name), nothing)
    meta === nothing && error("dataset '$name' not found")
    version = mpiio_version(ff)
    check_metadata(x, meta, version)
    file = parent(ff)
    offset = meta.offset_bytes :: Int
    chunks = meta.chunks :: Bool
    chunks && check_read_chunks(x, meta.process_dims, name)
    for u in collection(x)
        if chunks
            read_contiguous!(file, u; offset=offset, collective=collective, kw...)
        else
            read_discontiguous!(file, u; offset=offset, collective=collective, kw...)
        end
        offset += sizeof_global(u)
    end
    x
end

function check_metadata(x, meta, version)
    T = eltype_collection(x)
    file_eltype = meta.element_type
    if string(T) != file_eltype
        error("incompatible type of file and array: $file_eltype ≠ $T")
    end

    sz = (size_global(x, MemoryOrder())..., collection_size(x)...)
    file_dims = Tuple(meta.dims_memory) :: typeof(sz)
    if sz !== file_dims
        error("incompatible dimensions of dataset in file and array: $file_dims ≠ $sz")
    end

    file_sizeof = meta.size_bytes
    @assert sizeof_global(x) == file_sizeof

    file_bom = if version < v"0.9.4"
        # julia_endian_bom key didn't exist; assume ENDIAN_BOM
        ENDIAN_BOM
    else
        parse(typeof(ENDIAN_BOM), meta.julia_endian_bom)
    end

    if file_bom != ENDIAN_BOM
        error(
            "file was not written with the same native endianness of the current system." *
            " Reading a non-native endianness is not yet supported."
        )
    end

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
    to = timer(pencil(x))
    @timeit_debug to "Write MPI-IO discontiguous" begin
        set_view!(ff, x, offset; infokws...)
        A = parent(x)
        if collective
            MPI.File.write_all(ff, A)
        else
            MPI.File.write(ff, A)
        end
    end
    nothing
end

function read_discontiguous!(ff::MPI.FileHandle, x::PencilArray;
                             offset, collective, infokws...)
    to = timer(pencil(x))
    @timeit_debug to "Read MPI-IO discontiguous" begin
        set_view!(ff, x, offset; infokws...)
        A = parent(x)
        if collective
            MPI.File.read_all!(ff, A)
        else
            MPI.File.read!(ff, A)
        end
    end
end

function set_view!(ff, x::PencilArray, offset; infokws...)
    etype = MPI.Datatype(eltype(x))
    filetype = create_discontiguous_datatype(x, MemoryOrder())  # TODO cache datatype?
    datarep = "native"
    MPI.File.set_view!(ff, offset, etype, filetype, datarep; infokws...)
    nothing
end

function create_discontiguous_datatype(x::PencilArray, order)
    sizes = size_global(x, order)
    subsizes = size_local(x, order)
    offsets = map(r -> first(r) - 1, range_local(x, order))
    oldtype = MPI.Datatype(eltype(x))
    dtype = MPI.Types.create_subarray(sizes, subsizes, offsets, oldtype)
    MPI.Types.commit!(dtype)
    dtype
end

function write_contiguous(ff::MPI.FileHandle, x::PencilArray;
                          offset, collective, infokws...)
    to = timer(pencil(x))
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
    to = timer(pencil(x))
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
