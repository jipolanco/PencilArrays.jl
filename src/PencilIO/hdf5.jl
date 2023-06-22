# Note: the actual implementation is in the PencilArraysHDF5Ext.jl package extension.

export PHDF5Driver

"""
    PHDF5Driver(; fcpl = HDF5.FileCreateProperties(), fapl = HDF5.FileAccessProperties())

Parallel HDF5 driver using the HDF5.jl package.

HDF5 file creation and file access property lists may be specified via the
`fcpl` and `fapl` keyword arguments respectively.

Note that the MPIO file access property list does not need to be set, as this is
done automatically by this driver when the file is opened.
"""
struct PHDF5Driver{
        FileCreateProperties,  # type not known, since HDF5 hasn't been loaded at this point...
        FileAccessProperties,
    } <: ParallelIODriver
    fcpl :: FileCreateProperties
    fapl :: FileAccessProperties

    # "Private" constructor, called in package extension (PencilArraysHDF5Ext.jl).
    global _PHDF5Driver(a, b) = new{typeof(a), typeof(b)}(a, b)
end

"""
    hdf5_has_parallel() -> Bool

Returns `true` if the loaded HDF5 libraries support MPI-IO.

This is exactly the same as `HDF5.has_parallel()`, and is left here for
compatibility with previous versions.
"""
function hdf5_has_parallel end

"""
    open([f::Function], driver::PHDF5Driver, filename, comm::MPI.Comm; keywords...)

Open parallel file using the Parallel HDF5 driver.

See [`open(::ParallelIODriver)`](@ref) for common options for all drivers.

Driver-specific options may be passed via the `driver` argument. See
[`PHDF5Driver`](@ref) for details.
"""
function Base.open(::PHDF5Driver) end

"""
    setindex!(
        g::Union{HDF5.File, HDF5.Group}, x::MaybePencilArrayCollection,
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
  For instance, if the `dxpl_mpio = :collective` option is passed,
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
function Base.setindex!(::PHDF5Driver) end  # this is just for generating the documentation

"""
    read!(g::Union{HDF5.File, HDF5.Group}, x::MaybePencilArrayCollection,
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
function Base.read!(::PHDF5Driver) end
