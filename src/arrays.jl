"""
    PencilArray(pencil::Pencil, data::AbstractArray{T,N})

Create array wrapper with pencil decomposition information.

The array dimensions and element type must be consistent with those of the given
pencil.

!!! note "Index permutations"

    If the `Pencil` has an associated index permutation, then `data` must have
    its dimensions permuted accordingly (in *memory* order).

    Unlike `data`, the resulting `PencilArray` should be accessed with
    unpermuted indices (in *logical* order).

    ##### Example

    Suppose `pencil` has local dimensions `(10, 20, 30)` before permutation, and
    has an asociated permutation `(2, 3, 1)`.
    Then:
    ```julia
    data = zeros(20, 30, 10)       # parent array (with dimensions in memory order)

    u = PencilArray(pencil, data)  # wrapper with dimensions (10, 20, 30)
    @assert size(u) === (10, 20, 30)

    u[15, 25, 5]          # BoundsError (15 > 10 and 25 > 20)
    u[5, 15, 25]          # correct
    parent(u)[15, 25, 5]  # correct

    ```

!!! note "Extra dimensions"

    The data array can have one or more extra dimensions to the right (slow
    indices), which are not affected by index permutations.

    ##### Example

    ```julia
    dims = (20, 30, 10)
    PencilArray(pencil, zeros(dims...))        # works (scalar)
    PencilArray(pencil, zeros(dims..., 3))     # works (3-component vector)
    PencilArray(pencil, zeros(dims..., 4, 3))  # works (4×3 tensor)
    PencilArray(pencil, zeros(3, dims...))     # fails
    ```

---

    PencilArray{T}(undef, pencil::Pencil, [extra_dims...])

Allocate an uninitialised `PencilArray` that can hold data in the local pencil.

Extra dimensions, for instance representing vector components, can be specified.
These dimensions are added to the rightmost (slowest) indices of the resulting
array.

# Example
Suppose `pencil` has local dimensions `(20, 10, 30)`. Then:
```julia
PencilArray{Float64}(undef, pencil)        # array dimensions are (20, 10, 30)
PencilArray{Float64}(undef, pencil, 4, 3)  # array dimensions are (20, 10, 30, 4, 3)
```
"""
struct PencilArray{
        T,
        N,
        A <: AbstractArray{T,N},
        Np,  # number of "spatial" dimensions (i.e. dimensions of the Pencil)
        E,   # number of "extra" dimensions (= N - Np)
        P <: Pencil,
    } <: AbstractArray{T,N}
    pencil   :: P
    data     :: A
    space_dims :: Dims{Np}  # spatial dimensions in *logical* order
    extra_dims :: Dims{E}

    function PencilArray(pencil::Pencil{Np, Mp} where {Np, Mp},
                         data::AbstractArray{T, N}) where {T, N}
        P = typeof(pencil)
        A = typeof(data)
        Np = ndims(pencil)
        E = N - Np
        size_data = size(data)

        geom_dims = ntuple(n -> size_data[n], Np)  # = size_data[1:Np]
        extra_dims = ntuple(n -> size_data[Np + n], E)  # = size_data[Np+1:N]

        dims_local = size_local(pencil, MemoryOrder())

        if geom_dims !== dims_local
            throw(DimensionMismatch(
                "array has incorrect dimensions: $(size_data). " *
                "Local dimensions of pencil: $(dims_local)."))
        end

        iperm = inverse_permutation(get_permutation(pencil))
        space_dims = permute_indices(geom_dims, iperm)

        new{T, N, A, Np, E, P}(pencil, data, space_dims, extra_dims)
    end
end

function PencilArray{T}(init, pencil::Pencil, extra_dims::Vararg{Integer}) where {T}
    dims = (size_local(pencil, MemoryOrder())..., extra_dims...)
    PencilArray(pencil, Array{T}(init, dims))
end

@deprecate(
    PencilArray{T}(init, pencil::Pencil, extra_dims::NTuple) where {T},
    PencilArray{T}(init, pencil, extra_dims...),
)

"""
    PencilArrayCollection

`UnionAll` type describing a collection of [`PencilArray`](@ref)s.

Such a collection can be a tuple or an array of `PencilArray`s.

Collections are **by assumption** homogeneous: each array has the same
properties, and in particular, is associated to the same [`Pencil`](@ref)
configuration.

For convenience, certain operations defined for `PencilArray` are also defined
for `PencilArrayCollection`, and return the same value as for a single
`PencilArray`.
Some examples are [`pencil`](@ref), [`range_local`](@ref) and
[`get_comm`](@ref).

Also note that functions from `Base`, such as `size`, `ndims` and `eltype`, are **not**
overloaded for `PencilArrayCollection`, since they already have a definition
for tuples and arrays (and redefining them would be type piracy...).
"""
const PencilArrayCollection =
    Union{Tuple{Vararg{A}}, AbstractArray{A}} where {A <: PencilArray}

collection_size(x::Tuple{Vararg{<:PencilArray}}) = (length(x), )
collection_size(x::AbstractArray{<:PencilArray}) = size(x)
collection_size(::PencilArray) = ()

# This is convenient for iterating over one or more PencilArrays.
# A single PencilArray is treated as a "collection" of one array.
collection(x::PencilArrayCollection) = x
collection(x::PencilArray) = (x, )

const MaybePencilArrayCollection = Union{PencilArray, PencilArrayCollection}

function _apply(f::Function, x::PencilArrayCollection, args...; kwargs...)
    a = first(x)
    if !all(b -> pencil(a) === pencil(b), x)
        throw(ArgumentError("PencilArrayCollection is not homogeneous"))
    end
    f(a, args...; kwargs...)
end

"""
    size(x::PencilArray)

Return local dimensions of a `PencilArray` in logical order.
"""
Base.size(x::PencilArray) = (x.space_dims..., extra_dims(x)...)

"""
    size_local(x::PencilArray, [order = LogicalOrder()])
    size_local(x::PencilArrayCollection, [order = LogicalOrder()])

Local dimensions of the data held by the `PencilArray`.

If `order = LogicalOrder()`, this is the same as `size(x)`.

See also [`size_local(::Pencil)`](@ref).
"""
size_local(x::MaybePencilArrayCollection, args...; kwargs...) =
    (size_local(pencil(x), args...; kwargs...)..., extra_dims(x)...)

# TODO this won't work with extra_dims...
function Base.axes(x::PencilArray)
    iperm = inverse_permutation(get_permutation(x))
    permute_indices(axes(parent(x)), iperm)
end

function Base.similar(x::PencilArray, ::Type{S}, dims::Dims) where {S}
    perm = get_permutation(x)
    dims_perm = permute_indices(dims, perm)
    PencilArray(x.pencil, similar(x.data, S, dims_perm))
end

# Use same index style as the parent array.
Base.IndexStyle(::Type{<:PencilArray{T,N,A}} where {T,N}) where {A} =
    IndexStyle(A)

# Overload Base._sub2ind for converting from Cartesian to linear index.
# TODO this won't work with extra_dims...
@inline function Base._sub2ind(x::PencilArray, I...)
    # _sub2ind(axes(x), I...)  <- default implementation for AbstractArray
    J = permute_indices(I, get_permutation(x))
    Base._sub2ind(parent(x), J...)
end

# Linear indexing
@propagate_inbounds @inline Base.getindex(x::PencilArray, i::Integer) =
    x.data[i]

@propagate_inbounds @inline Base.setindex!(x::PencilArray, v, i::Integer) =
    x.data[i] = v

# Cartesian indexing: assume input indices are unpermuted, and permute them.
# (This is similar to the implementation of PermutedDimsArray.)
@propagate_inbounds @inline Base.getindex(
        x::PencilArray{T,N}, I::Vararg{Int,N}) where {T,N} =
    x.data[_genperm(x, I)...]

@propagate_inbounds @inline Base.setindex!(
        x::PencilArray{T,N}, v, I::Vararg{Int,N}) where {T,N} =
    x.data[_genperm(x, I)...] = v

@inline function _genperm(x::PencilArray{T,N}, I::NTuple{N,Int}) where {T,N}
    # Split "spatial" and "extra" indices.
    M = ndims_space(x)
    E = ndims_extra(x)
    @assert M + E === N
    J = ntuple(n -> I[n], Val(M))
    K = ntuple(n -> I[M + n], Val(E))
    perm = get_permutation(x)
    (permute_indices(J, perm)..., K...)
end

@inline _genperm(x::PencilArray, I::CartesianIndex) =
    CartesianIndex(_genperm(x, Tuple(I)))

"""
    pencil(x::PencilArray)

Return decomposition configuration associated to a `PencilArray`.
"""
pencil(x::PencilArray) = x.pencil
pencil(x::PencilArrayCollection) = _apply(pencil, x)

"""
    parent(x::PencilArray)

Return array wrapped by a `PencilArray`.
"""
Base.parent(x::PencilArray) = x.data

# This enables aliasing detection (e.g. using Base.mightalias) on PencilArrays.
Base.dataids(x::PencilArray) = Base.dataids(parent(x))

"""
    pointer(x::PencilArray)

Return pointer to the start of the underlying data.

Use with caution: this may not make a lot of sense if the underlying data is not
contiguous or strided (e.g. if the `PencilArray` is wrapping a non-strided
`SubArray`).
"""
Base.pointer(x::PencilArray) = pointer(parent(x))

"""
    ndims_extra(x::PencilArray)
    ndims_extra(x::PencilArrayCollection)

Number of "extra" dimensions associated to `PencilArray`.

These are the dimensions that are not associated to the domain geometry.
For instance, they may correspond to vector or tensor components.

These dimensions correspond to the rightmost indices of the array.

The total number of dimensions of a `PencilArray` is given by:

    ndims(x) == ndims_space(x) + ndims_extra(x)

"""
ndims_extra(x::MaybePencilArrayCollection) = length(extra_dims(x))

"""
    ndims_space(x::PencilArray)
    ndims_space(x::PencilArrayCollection)

Number of dimensions associated to the domain geometry.

These dimensions correspond to the leftmost indices of the array.

The total number of dimensions of a `PencilArray` is given by:

    ndims(x) == ndims_space(x) + ndims_extra(x)

"""
ndims_space(x::PencilArray) = ndims(x) - ndims_extra(x)
ndims_space(x::PencilArrayCollection) = _apply(ndims_space, x)

"""
    extra_dims(x::PencilArray)
    extra_dims(x::PencilArrayCollection)

Return tuple with size of "extra" dimensions of `PencilArray`.
"""
extra_dims(x::PencilArray) = x.extra_dims
extra_dims(x::PencilArrayCollection) = _apply(extra_dims, x)

"""
    size_global(x::PencilArray, [order = LogicalOrder()])
    size_global(x::PencilArrayCollection, [order = LogicalOrder()])

Global dimensions associated to the given array.

By default, the logical dimensions of the dataset are returned.

See also [`size_global(::Pencil)`](@ref).
"""
size_global(x::MaybePencilArrayCollection, args...; kw...) =
    (size_global(pencil(x), args...; kw...)..., extra_dims(x)...)

"""
    sizeof_global(x::PencilArray)
    sizeof_global(x::PencilArrayCollection)

Global size of array in bytes.
"""
sizeof_global(x::PencilArray) = prod(size_global(x)) * sizeof(eltype(x))
sizeof_global(x::PencilArrayCollection) = sum(sizeof_global.(x))

"""
    range_local(x::PencilArray, [order = LogicalOrder()])
    range_local(x::PencilArrayCollection, [order = LogicalOrder()])

Local data range held by the `PencilArray`.

By default the dimensions are returned in logical order.
"""
range_local(x::MaybePencilArrayCollection, args...; kw...) =
    (range_local(pencil(x), args...; kw...)..., Base.OneTo.(extra_dims(x))...)

"""
    range_remote(x::PencilArray, coords, [order = LogicalOrder()])
    range_remote(x::PencilArrayCollection, coords, [order = LogicalOrder()])

Get data range held by the `PencilArray` in a given MPI process.

The location of the MPI process in the topology is determined by the `coords`
argument, which can be given as a linear or Cartesian index.

See [`range_remote(::Pencil, ...)`](@ref range_remote(::Pencil, ::Integer,
::LogicalOrder)) variant for details.
"""
range_remote(x::MaybePencilArrayCollection, args...) =
    (range_remote(pencil(x), args...)..., Base.OneTo.(extra_dims(x))...)

"""
    get_comm(x::PencilArray)
    get_comm(x::PencilArrayCollection)

Get MPI communicator associated to a pencil-distributed array.
"""
get_comm(x::MaybePencilArrayCollection) = get_comm(pencil(x))

"""
    get_permutation(x::PencilArray)
    get_permutation(x::PencilArrayCollection)

Get index permutation associated to the given `PencilArray`.

Returns `NoPermutation()` if there is no associated permutation.
"""
function get_permutation(x::MaybePencilArrayCollection)
    perm = get_permutation(pencil(x))

    # Account for extra dimensions.
    E = ndims_extra(x)
    append_to_permutation(perm, Val(E))
end

"""
    topology(x::PencilArray)
    topology(x::PencilArrayCollection)

Get [`MPITopology`](@ref) associated to a `PencilArray`.
"""
topology(x::MaybePencilArrayCollection) = topology(pencil(x))

"""
    gather(x::PencilArray, [root::Integer=0])

Gather data from all MPI processes into one (big) array.

Data is received by the `root` process.

Returns the full array on the `root` process, and `nothing` on the other
processes.

This can be useful for testing, but it shouldn't be used with very large
datasets!
"""
function gather(x::PencilArray{T,N}, root::Integer=0) where {T, N}
    timer = get_timer(pencil(x))

    @timeit_debug timer "gather" begin

    # TODO reduce allocations! see `transpose_impl!`
    comm = get_comm(x)
    rank = MPI.Comm_rank(comm)
    mpi_tag = 42
    pen = pencil(x)
    extra_dims = x.extra_dims

    # Each process sends its data to the root process.
    # If the local indices are permuted, the permutation is reverted before
    # sending the data.
    data = let perm = pen.perm
        if is_identity_permutation(perm)
            x.data
        else
            # Apply inverse permutation.
            invperm = inverse_permutation(perm)
            p = append_to_permutation(invperm, Val(length(extra_dims)))
            permutedims(x.data, Tuple(p))  # creates copy!
        end
    end

    if rank != root
        # Wait for data to be sent, then return.
        # NOTE: When `data` is a ReshapedArray, I can't pass it directly to
        # MPI.Isend, because Base.cconvert(MPIPtr, ::ReshapedArray) is not
        # defined.
        # (Maybe it works in the current master of MPI.jl?)
        buf = data isa Base.ReshapedArray ? parent(data) : data
        send_req = MPI.Isend(buf, root, mpi_tag, comm)
        MPI.Wait!(send_req)
        return nothing
    end

    # Receive data (root only).
    topo = pen.topology
    Nproc = length(topo)
    recv = Vector{Array{T,N}}(undef, Nproc)
    recv_req = Vector{MPI.Request}(undef, Nproc)

    root_index = -1

    for n = 1:Nproc
        # Global data range that I will receive from process n.
        rrange = pen.axes_all[n]
        rdims = length.(rrange)

        src_rank = topo.ranks[n]  # actual rank of sending process
        if src_rank == root
            root_index = n
            recv_req[n] = MPI.REQUEST_NULL
        else
            # TODO avoid allocation?
            recv[n] = Array{T,N}(undef, rdims..., extra_dims...)
            recv_req[n] = MPI.Irecv!(recv[n], src_rank, mpi_tag, comm)
        end
    end

    # Unpack data.
    dest = Array{T,N}(undef, size_global(x))

    # Copy local data.
    colons_extra_dims = ntuple(n -> Colon(), Val(length(extra_dims)))
    dest[pen.axes_local..., colons_extra_dims...] .= data

    # Copy remote data.
    for m = 2:Nproc
        n, status = MPI.Waitany!(recv_req)
        rrange = pen.axes_all[n]
        dest[rrange..., colons_extra_dims...] .= recv[n]
    end

    end  # @timeit_debug

    dest
end
