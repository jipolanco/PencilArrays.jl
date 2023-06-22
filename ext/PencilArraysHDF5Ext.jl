module PencilArraysHDF5Ext

using MPI
using HDF5
using PencilArrays
using PencilArrays.PencilIO
using PencilArrays: MaybePencilArrayCollection, collection_size
using StaticArrays: SVector
using TimerOutputs

function PencilIO.PHDF5Driver(;
        fcpl = HDF5.FileCreateProperties(),
        fapl = HDF5.FileAccessProperties(),
    )
    # We set fapl.fclose_degree if it hasn't been explicitly set.
    if !_is_set(fapl, Val(:fclose_degree))
        # This is the default in HDF5.jl -- makes sense due to GC.
        fapl.fclose_degree = :strong
    end
    PencilIO._PHDF5Driver(fcpl, fapl)
end

# TODO Is there a better way to check if fapl.fclose_degree has already
# been set??
function _is_set(fapl::HDF5.FileAccessProperties, ::Val{:fclose_degree})
    id = fapl.id
    degree = Ref{Cint}()
    status = ccall(
        (:H5Pget_fclose_degree, HDF5.API.libhdf5), HDF5.API.herr_t,
        (HDF5.API.hid_t, Ref{Cint}), id, degree)
    # A negative value means failure, which we interpret here as meaning that
    # "fclose_degree" has not been set.
    status ≥ 0
end

PencilIO.hdf5_has_parallel() = HDF5.has_parallel()

function keywords_to_h5open(; kws...)
    flags, other_kws = PencilIO.keywords_to_open(; kws...)
    (
        flags.read,
        flags.write,
        flags.create,
        flags.truncate,
        flags.append,
    ), other_kws
end

function Base.open(D::PHDF5Driver, filename::AbstractString, comm::MPI.Comm; kw...)
    mode_args, other_kws = keywords_to_h5open(; kw...)
    info = MPI.Info(other_kws...)
    fcpl = D.fcpl
    fapl = D.fapl
    mpio = HDF5.Drivers.MPIO(comm, info)
    HDF5.Drivers.set_driver!(fapl, mpio)  # fails if no parallel support
    swmr = false

    # The code below is adapted from h5open in HDF5.jl v0.15
    # TODO propose alternative h5open for HDF5.jl, taking keyword arguments `read`, `write`, ...
    # Then we wouldn't need to copy code from HDF5.jl...
    rd, wr, cr, tr, ff = mode_args
    if ff && !wr
        error("HDF5 does not support appending without writing")
    end

    fid = if cr && (tr || !isfile(filename))
        flag = swmr ? HDF5.API.H5F_ACC_TRUNC | HDF5.API.H5F_ACC_SWMR_WRITE :
                      HDF5.API.H5F_ACC_TRUNC
        HDF5.API.h5f_create(filename, flag, fcpl, fapl)
    else
        HDF5.ishdf5(filename) ||
            error("unable to determine if $filename is accessible in the HDF5 format (file may not exist)")
        flag = if wr
            swmr ? HDF5.API.H5F_ACC_RDWR | HDF5.API.H5F_ACC_SWMR_WRITE :
                   HDF5.API.H5F_ACC_RDWR
        else
            swmr ? HDF5.API.H5F_ACC_RDONLY | HDF5.API.H5F_ACC_SWMR_READ :
                   HDF5.API.H5F_ACC_RDONLY
        end
        HDF5.API.h5f_open(filename, flag, fapl)
    end

    close(fapl)
    close(fcpl)

    HDF5.File(fid, filename)
end

function Base.setindex!(
        g::Union{HDF5.File, HDF5.Group}, x::MaybePencilArrayCollection,
        name::AbstractString;
        chunks=false, collective=true, prop_pairs...,
    )
    to = timer(pencil(x))

    @timeit_debug to "Write HDF5" begin

    check_phdf5_file(g, x)

    # Add extra property lists if required by keyword args.
    # TODO avoid using Dict?
    props = Dict{Symbol,Any}(pairs(prop_pairs))

    if chunks && !haskey(prop_pairs, :chunk)
        chunk = h5_chunk_size(x, MemoryOrder())
        props[:chunk] = chunk
    end

    if collective && !haskey(prop_pairs, :dxpl_mpio)
        props[:dxpl_mpio] = :collective
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
    meta = PencilIO.metadata(x)
    for (name, val) in pairs(meta)
        dset[string(name)] = to_hdf5(val)
    end
    dset
end

to_hdf5(val) = val
to_hdf5(val::Tuple{}) = false  # empty tuple
to_hdf5(val::Tuple) = SVector(val)
to_hdf5(::Nothing) = false

function Base.read!(g::Union{HDF5.File, HDF5.Group}, x::MaybePencilArrayCollection,
                    name::AbstractString; collective=true, prop_pairs...)
    to = timer(pencil(x))

    @timeit_debug to "Read HDF5" begin

    dapl = HDF5.DatasetAccessProperties(; prop_pairs...)
    dxpl = HDF5.DatasetTransferProperties(; prop_pairs...)

    # Add extra property lists if required by keyword args.
    if collective && !haskey(prop_pairs, :dxpl_mpio)
        dxpl.dxpl_mpio = :collective
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
    fapl = HDF5.get_access_properties(HDF5.file(g))
    driver = HDF5.Drivers.get_driver(fapl)
    if driver isa HDF5.Drivers.MPIO
        comm = driver.comm
        if MPI.Comm_compare(comm, get_comm(x)) ∉ (MPI.IDENT, MPI.CONGRUENT)
            throw(ArgumentError(
                "incompatible MPI communicators of HDF5 file and PencilArray"
            ))
        end
    else
        error("HDF5 file was not opened with the MPIO driver")
    end
    close(fapl)
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
        HDF5.API.h5d_read(dset.id, memtype.id, memspace.id, dsel_id, dset.xfer, u)
    finally
        close(memtype)
        close(memspace)
        HDF5.API.h5s_close(dsel_id)
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

end
