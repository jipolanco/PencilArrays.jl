using .HDF5
import Libdl

export PHDF5Driver

"""
    PHDF5Driver(; fcpl, fapl)

Parallel HDF5 driver using the HDF5.jl package.

HDF5 file creation and file access property lists may be specified via the
`fcpl` and `fapl` keyword arguments respectively.

Note that the MPIO file access property list does not need to be set, as this is
done automatically by this driver when the file is opened.
"""
struct PHDF5Driver <: ParallelIODriver
    fcpl :: HDF5.Properties
    fapl :: HDF5.Properties
    function PHDF5Driver(;
            fcpl = HDF5.DEFAULT_PROPERTIES,
            fapl = HDF5.DEFAULT_PROPERTIES,
        )
        check_hdf5_parallel()
        if fcpl == HDF5.DEFAULT_PROPERTIES
            fcpl = create_property(HDF5.H5P_FILE_CREATE)
        end
        if fapl == HDF5.DEFAULT_PROPERTIES
            fapl = create_property(HDF5.H5P_FILE_ACCESS)
            # This is the default in HDF5.jl -- makes sense due to GC
            fapl[:fclose_degree] = HDF5.H5F_CLOSE_STRONG
        end
        new(fcpl, fapl)
    end
end

const HDF5FileOrGroup = Union{HDF5.File, HDF5.Group}

const H5MPIHandle = let csize = sizeof(MPI.MPI_Comm)
    @assert csize in (4, 8)
    csize === 4 ? HDF5.Hmpih32 : HDF5.Hmpih64
end

mpi_to_h5_handle(obj::Union{MPI.Comm, MPI.Info}) =
    reinterpret(H5MPIHandle, obj.val)

function h5_to_mpi_handle(
        ::Type{T}, h::H5MPIHandle) where {T <: Union{MPI.Comm, MPI.Info}}
    T(reinterpret(MPI.MPI_Comm, h))
end

const _HAS_PARALLEL_HDF5 = Libdl.dlopen(HDF5.libhdf5) do lib
    Libdl.dlsym(lib, :H5Pget_fapl_mpio, throw_error=false) !== nothing
end

"""
    hdf5_has_parallel() -> Bool

Returns `true` if the loaded HDF5 libraries support MPI-IO.
"""
hdf5_has_parallel() = _HAS_PARALLEL_HDF5

function check_hdf5_parallel()
    hdf5_has_parallel() && return
    error(
        "HDF5.jl has no parallel support." *
        " Make sure that you're using MPI-enabled HDF5 libraries, and that" *
        " MPI was loaded before HDF5." *
        " See https://jipolanco.github.io/PencilArrays.jl/latest/PencilIO/#Setting-up-Parallel-HDF5-1" *
        " for details."
    )
end

function keywords_to_h5open(; kws...)
    flags, other_kws = keywords_to_open(; kws...)
    (
        flags.read,
        flags.write,
        flags.create,
        flags.truncate,
        flags.append,
    ), other_kws
end

"""
    open([f::Function], driver::PHDF5Driver, filename, comm::MPI.Comm; keywords...)

Open parallel file using the Parallel HDF5 driver.

See [`open(::ParallelIODriver)`](@ref) for common options for all drivers.

Driver-specific options may be passed via the `driver` argument. See
[`PHDF5Driver`](@ref) for details.
"""
function Base.open(::PHDF5Driver) end

function Base.open(D::PHDF5Driver, filename::AbstractString, comm::MPI.Comm; kw...)
    mode_args, other_kws = keywords_to_h5open(; kw...)
    info = MPI.Info(other_kws...)
    D.fapl[:fapl_mpio] = mpi_to_h5_handle.((comm, info))
    # TODO avoid using unexported function
    HDF5._h5open(string(filename), mode_args..., D.fcpl, D.fapl)
end

"""
    setindex!(
        g::Union{HDF5File,HDF5Group}, x::MaybePencilArrayCollection,
        name::AbstractString; chunks = false, collective = true, prop_lists...,
    )

Write [`PencilArray`](@ref) or [`PencilArrayCollection`](@ref) to parallel HDF5
file.

For performance reasons, the memory layout of the data is conserved. In other
words, if the dimensions of a `PencilArray` are permuted in memory, then the
data is written in permuted form.

In the case of a `PencilArrayCollection`, each array of the collection is written
as a single component of a higher-dimension dataset.

# Optional arguments

- if `chunks = true`, data is written in chunks, with roughly one chunk
  per MPI process. This may (or may not) improve performance in parallel
  filesystems.

- if `collective = true`, the dataset is written collectivelly. This is
  usually recommended for performance.

- additional property lists may be specified by key-value pairs in
  `prop_lists`, following the [HDF5.jl
  syntax](https://juliaio.github.io/HDF5.jl/stable/#Passing-parameters).
  These property lists take precedence over keyword arguments.
  For instance, if the `dxpl_mpio = HDF5.H5FD_MPIO_COLLECTIVE` option is passed,
  then the value of the `collective` argument is ignored.

# Property lists

Property lists are passed to
[`h5d_create`](https://portal.hdfgroup.org/display/HDF5/H5D_CREATE2)
and [`h5d_write`](https://portal.hdfgroup.org/display/HDF5/H5D_WRITE).
The following property types are recognised:
- [link creation properties](https://portal.hdfgroup.org/display/HDF5/Attribute+and+Link+Creation+Properties),
- [dataset creation properties](https://portal.hdfgroup.org/display/HDF5/Dataset+Creation+Properties),
- [dataset access properties](https://portal.hdfgroup.org/display/HDF5/Dataset+Access+Properties),
- [dataset transfer properties](https://portal.hdfgroup.org/display/HDF5/Dataset+Transfer+Properties).

# Example

Open a parallel HDF5 file and write some `PencilArray`s to the file:

```julia
pencil = Pencil(#= ... =#)
u = PencilArray{Float64}(undef, pencil)
v = similar(u)

# [fill the arrays with interesting values...]

comm = get_comm(u)

open(PHDF5Driver(), "filename.h5", comm, write=true) do ff
    ff["u", chunks=true] = u
    ff["uv"] = (u, v)  # this is a two-component PencilArrayCollection (assuming equal dimensions of `u` and `v`)
end
```

"""
function Base.setindex!(g::HDF5FileOrGroup, x::MaybePencilArrayCollection,
                        name::AbstractString; kws...)
    setindex!(g, x, string(name); kws...)
end

# This definition is needed to avoid method ambiguity error.
# This is because HDF5.jl defines setindex! for String, not AbstractString.
function Base.setindex!(
        g::HDF5FileOrGroup, x::MaybePencilArrayCollection,
        name::String; chunks=false, collective=true, prop_pairs...,
    )
    to = get_timer(pencil(x))

    @timeit_debug to "Write HDF5" begin

    check_phdf5_file(g, x)

    # Add extra property lists if required by keyword args.
    props = Dict{Symbol,Any}(pairs(prop_pairs))

    if chunks && !haskey(prop_pairs, :chunk)
        chunk = h5_chunk_size(x, MemoryOrder())
        props[:chunk] = chunk
    end

    if collective && !haskey(prop_pairs, :dxpl_mpio)
        props[:dxpl_mpio] = HDF5.H5FD_MPIO_COLLECTIVE
    end

    dims_global = h5_dataspace_dims(x)
    @timeit_debug to "create dataset" dset =
        create_dataset(g, name, h5_datatype(x), dataspace(dims_global); props...)
    inds = range_local(x, MemoryOrder())
    @timeit_debug to "write data" to_hdf5(dset, x, inds)
    @timeit_debug to "write metadata" write_metadata(dset, x)

    end

    x
end

# Write metadata as HDF5 attributes attached to a dataset.
# Note that this is a collective operation (all processes must call this).
function write_metadata(dset::HDF5.Dataset, x)
    meta = metadata(x)
    for (name, val) in pairs(meta)
        dset[string(name)] = to_hdf5(val)
    end
    dset
end

to_hdf5(val) = val
to_hdf5(val::Tuple{}) = false  # empty tuple
to_hdf5(val::Tuple) = SVector(val)
to_hdf5(::Nothing) = false

"""
    read!(g::Union{HDF5File,HDF5Group}, x::MaybePencilArrayCollection,
          name::AbstractString; collective=true, prop_lists...)

Read [`PencilArray`](@ref) or [`PencilArrayCollection`](@ref) from parallel HDF5
file.

See [`setindex!`](@ref) for details on optional arguments.

# Property lists

Property lists are passed to
[`h5d_open`](https://portal.hdfgroup.org/display/HDF5/H5D_OPEN2)
and [`h5d_read`](https://portal.hdfgroup.org/display/HDF5/H5D_READ).
The following property types are recognised:
- [dataset access properties](https://portal.hdfgroup.org/display/HDF5/Dataset+Access+Properties),
- [dataset transfer properties](https://portal.hdfgroup.org/display/HDF5/Dataset+Transfer+Properties).

# Example

Open a parallel HDF5 file and read some `PencilArray`s:

```julia
pencil = Pencil(#= ... =#)
u = PencilArray{Float64}(undef, pencil)
v = similar(u)

comm = get_comm(u)
info = MPI.Info()

open(PHDF5Driver(), "filename.h5", comm, read=true) do ff
    read!(ff, u, "u")
    read!(ff, (u, v), "uv")
end
```
"""
function Base.read!(g::HDF5FileOrGroup, x::MaybePencilArrayCollection,
                    name::AbstractString; collective=true, prop_pairs...)
    to = get_timer(pencil(x))

    @timeit_debug to "Read HDF5" begin

    dapl = create_property(HDF5.H5P_DATASET_ACCESS; prop_pairs...)
    dxpl = create_property(HDF5.H5P_DATASET_XFER; prop_pairs...)

    # Add extra property lists if required by keyword args.
    if collective && !haskey(prop_pairs, :dxpl_mpio)
        HDF5.h5p_set_dxpl_mpio(dxpl.id, HDF5.H5FD_MPIO_COLLECTIVE)
    end

    dims_global = h5_dataspace_dims(x)
    @timeit_debug to "open dataset" dset = open_dataset(g, string(name), dapl, dxpl)
    check_phdf5_file(parent(dset), x)

    if dims_global != size(dset)
        throw(DimensionMismatch(
            "incompatible dimensions of HDF5 dataset and PencilArray"))
    end

    inds = range_local(x, MemoryOrder())
    @timeit_debug to "read data" from_hdf5!(dset, x, inds)

    end

    x
end

function check_phdf5_file(g, x)
    check_hdf5_parallel()

    plist_id = HDF5.h5f_get_access_plist(HDF5.file(g))
    plist = HDF5.Properties(plist_id, HDF5.H5P_FILE_ACCESS)

    # Get HDF5 ids of MPIO driver and of the actual driver being used.
    driver_mpio = ccall((:H5FD_mpio_init, HDF5.libhdf5), HDF5.hid_t, ())
    driver = HDF5.h5p_get_driver(plist)
    if driver !== driver_mpio
        error("HDF5 file was not opened with the MPIO driver")
    end

    comm, info = _h5p_get_fapl_mpio(plist)
    if isdefined(MPI, :Comm_compare)  # requires recent MPI.jl
        if MPI.Comm_compare(comm, get_comm(x)) ∉ (MPI.IDENT, MPI.CONGRUENT)
            throw(ArgumentError(
                "incompatible MPI communicators of HDF5 file and PencilArray"
            ))
        end
    end

    close(plist)
    nothing
end

to_hdf5(dset, x::PencilArray, inds) = dset[inds...] = parent(x)

function from_hdf5!(dset, x::PencilArray, inds)
    u = parent(x)

    if stride(u, 1) != 1
        u .= dset[inds...]  # short and easy version (but allocates!)
        return x
    end

    # The following is adapted from one of the _getindex() in HDF5.jl.
    HDF5Scalar = HDF5.ScalarType
    T = eltype(x)
    if !(T <: Union{HDF5Scalar, Complex{<:HDF5Scalar}})
        error("Dataset indexing (hyperslab) is available only for bits types")
    end

    dsel_id = HDF5.hyperslab(dset, inds...)
    memtype = HDF5.datatype(u)
    memspace = HDF5.dataspace(u)

    try
        # This only works for stride-1 arrays.
        HDF5.h5d_read(dset.id, memtype.id, memspace.id, dsel_id, dset.xfer, u)
    finally
        close(memtype)
        close(memspace)
        HDF5.h5s_close(dsel_id)
    end

    x
end

# Define variants for collections.
for func in (:from_hdf5!, :to_hdf5)
    @eval function $func(dset, col::PencilArrayCollection, inds_in)
        for I in CartesianIndices(collection_size(col))
            inds = (inds_in..., Tuple(I)...)
            $func(dset, col[I], inds)
        end
    end
end

h5_datatype(x::PencilArray) = datatype(eltype(x))
h5_datatype(x::PencilArrayCollection) = h5_datatype(first(x))

h5_dataspace_dims(x::PencilArray) = size_global(x, MemoryOrder())
h5_dataspace_dims(x::PencilArrayCollection) =
    (h5_dataspace_dims(first(x))..., collection_size(x)...)

function h5_chunk_size(x::PencilArray, order = MemoryOrder())
    # Determine chunk size for writing to HDF5 dataset.
    # The idea is that each process writes to a single separate chunk of the
    # dataset, of size `dims_local`.
    # This only works if the data is ideally balanced among processes, i.e. if
    # the local dimensions of the dataset are the same for all processes.
    dims_local = size_local(x, order)

    # In the general case that the data is not well balanced, we take the
    # minimum size along each dimension.
    chunk = MPI.Allreduce(collect(dims_local), min, get_comm(x))

    N = ndims(x)
    @assert length(chunk) == N
    ntuple(d -> chunk[d], Val(N))
end

h5_chunk_size(x::PencilArrayCollection, args...) =
    (h5_chunk_size(first(x), args...)..., collection_size(x)...)

function _h5p_get_fapl_mpio(fapl_id)
    h5comm, h5info = HDF5.h5p_get_fapl_mpio(fapl_id, H5MPIHandle)
    comm = h5_to_mpi_handle(MPI.Comm, h5comm)
    info = h5_to_mpi_handle(MPI.Info, h5info)
    comm, info
end
