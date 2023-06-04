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
    @assert size_local(u) === (10, 20, 30)

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

    PencilArray{T}(undef, pencil::Pencil, [extra_dims...]; singleton = ())

Allocate an uninitialised `PencilArray` that can hold data in the local pencil.

By default, the array has the same dimensions as the local part of the pencil.

See also [`partition`](@ref) for an alternative way of constructing `PencilArray`s.

# Optional arguments

## Extra dimensions

Extra dimensions can be specified, which will be added to the rightmost
(slowest) indices of the resulting arrays.
These dimensions might represent things like different vector components in
physical applications.

## Constant (or singleton) dimensions

It is possible to specify that some of the dimensions should have size 1 --
instead of having the same size as the pencil.
This can be convenient for the purposes of describing physical fields that are
constant along one of the dimensions.
The resulting array can be naturally broadcasted with other `PencilArray`s that
hold data in the same pencil.

To specify one or more singleton dimensions, set the optional `singleton`
keyword argument to a dimension (e.g. `singleton = 2`) or a list of dimensions
(e.g. `singleton = (1, 3)`).
See below for more examples.

# Examples

Suppose `pencil` has local dimensions `(20, 10, 30)`. Then:

```julia
PencilArray{Float64}(undef, pencil)                 # array dimensions are (20, 10, 30)
PencilArray{Float64}(undef, pencil; singleton = 1)  # array dimensions are ( 1, 10, 30)
PencilArray{Float64}(undef, pencil, 4, 3)           # array dimensions are (20, 10, 30, 4, 3)
PencilArray{Float64}(undef, pencil, 4, 3; singleton = (1, 3))  # array dimensions are (1, 10, 1, 4, 3)
```

More examples:

```jldoctest
julia> pen = Pencil((20, 10, 12), MPI.COMM_WORLD);

julia> u = PencilArray{Float64}(undef, pen);

julia> v = PencilArray{Int}(undef, pen; singleton = 3);

julia> summary(u)
"20×10×12 PencilArray{Float64, 3}(::Pencil{3, 2, NoPermutation, Array})"

julia> summary(v)
"20×10×1 PencilArray{Int64, 3}(::Pencil{3, 2, NoPermutation, Array})"

julia> u .= 2.1; v .= 42; w = u .+ v; summary(w)
"20×10×12 PencilArray{Float64, 3}(::Pencil{3, 2, NoPermutation, Array})"

julia> extrema(w)
(44.1, 44.1)

julia> PencilArray{Float64}(undef, pen, 4, 3) |> summary
"20×10×12×4×3 PencilArray{Float64, 5}(::Pencil{3, 2, NoPermutation, Array})"
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
    dims_extra  :: Dims{E}   # optional rightmost dimensions, which are never distributed

    # Dimensions of global array (in logical order).
    # Note that this only includes the first Np dimensions, i.e. the "physical"
    # dimensions that can be distributed.
    # These are generally the same as size_global(pencil), except for singleton
    # dimensions, in which case they are 1.
    dims_global :: Dims{Np}

    # NOTE: this constructor is not to be used directly!!!
    function PencilArray(
            pencil::Pencil{Np}, data::AbstractArray{T, N},
            dims_global::Dims{Np} = size_global(pencil, LogicalOrder()),
        ) where {T, Np, N}
        _check_compatible(pencil, data)

        E = N - Np
        dims_data = size(data)

        perm = permutation(pencil)
        dims_space_mem = ntuple(n -> dims_data[n], Np)  # = dims_data[1:Np]
        dims_pencil_mem = size_local(pencil, MemoryOrder())
        dims_glob_mem = perm * dims_global

        # We compare dimensions in memory order (as they are stored in `data`)
        dims_are_ok = all(
                zip(dims_space_mem, dims_pencil_mem, dims_glob_mem)
            ) do (ndata, nl, ng)
            # A dimension size is "correct" if it's either equal to the local
            # pencil size, or equal to 1 (for singleton array dimensions).
            # We expect a singleton dimension if the given *global* size is 1.
            expect_singleton = ng == 1
            expect_singleton ? (ndata == 1) : (ndata == nl && ndata ≤ ng)
        end

        if !dims_are_ok
            throw(DimensionMismatch(
                "array has incorrect dimensions: $(dims_data). " *
                "Local dimensions of pencil (in memory order): $(dims_pencil_mem)."
            ))
        end

        dims_extra = ntuple(n -> dims_data[Np + n], E)  # = dims_data[Np+1:N]
        P = typeof(pencil)

        new{T, N, typeof(data), Np, E, P}(pencil, data, dims_extra, dims_global)
    end
end

@inline _check_compatible(p::Pencil, u) = _check_compatible(typeof_array(p), u)

@inline function _check_compatible(::Type{A}, u, ubase = u) where {A}
    typeof(u) <: A && return nothing
    up = parent(u)
    typeof(up) === typeof(u) && throw(ArgumentError(
        "type of data array ($(typeof(ubase))) is not compatible with expected array type ($A)"
    ))
    _check_compatible(A, up, ubase)
end

_apply_singleton(dims::NTuple, ::Tuple{}) = dims

function _apply_singleton(dims::NTuple, singleton)
    T = eltype(dims)
    for i ∈ singleton
        dims = Base.setindex(dims, one(T), i)
    end
    dims
end

function PencilArray{T}(
        init, pencil::Pencil, extra_dims::Vararg{Integer};
        singleton = (),
    ) where {T}
    dims_log = let dims = size_local(pencil, LogicalOrder())
        _apply_singleton(dims, singleton)
    end
    perm = permutation(pencil)
    dims_mem = perm * dims_log
    dims = (dims_mem..., extra_dims...)
    dims_global = let dims = size_global(pencil, LogicalOrder())
        _apply_singleton(dims, singleton)
    end
    A = typeof_array(pencil)
    PencilArray(pencil, A{T}(init, dims), dims_global)
end

# Treat PencilArray similarly to other wrapper types.
# https://github.com/JuliaGPU/Adapt.jl/blob/master/src/wrappers.jl
function Adapt.adapt_structure(to, u::PencilArray)
    A = typeof_array(to)
    p = similar(pencil(u), A)  # create Pencil with possibly different array type
    PencilArray(p, Adapt.adapt(to, parent(u)))
end

pencil_type(::Type{PencilArray{T,N,A,M,E,P}}) where {T,N,A,M,E,P} = P

# This is called by `summary`.
function Base.showarg(io::IO, u::PencilArray, toplevel)
    toplevel || print(io, "::")
    print(io, nameof(typeof(u)), '{', eltype(u), ", ", ndims(u), '}')
    if toplevel
        print(io, '(')
        Base.showarg(io, pencil(u), false)
        print(io, ')')
    end
    nothing
end

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

collection_size(x::Tuple{Vararg{PencilArray}}) = (length(x), )
collection_size(x::AbstractArray{<:PencilArray}) = size(x)
collection_size(::PencilArray) = ()

# This is convenient for iterating over one or more PencilArrays.
# A single PencilArray is treated as a "collection" of one array.
collection(x::PencilArrayCollection) = x
collection(x::PencilArray) = (x, )

const MaybePencilArrayCollection = Union{PencilArray, PencilArrayCollection}

function _only(f::F, x::PencilArrayCollection, args...; kwargs...) where {F}
    a = first(x)
    if !all(b -> pencil(a) === pencil(b), x)
        throw(ArgumentError("PencilArrayCollection is not homogeneous"))
    end
    f(a, args...; kwargs...)
end

Base.axes(x::PencilArray) = permutation(x) \ axes(parent(x))

"""
    similar(x::PencilArray, [element_type=eltype(x)], [dims]; [singleton = ()])

Returns an array similar to `x`.

The actual type of the returned array depends on whether `dims` is passed:

1. if `dims` is *not* passed, then a `PencilArray` of same dimensions of `x` is
   returned.

2. otherwise, an array similar to that wrapped by `x` (typically a regular
   `Array`) is returned, with the chosen dimensions.

The optional `singleton` argument is only allowed in case 1.
See the [`PencilArray`](@ref) constructor for its meaning.

# Examples

```jldoctest
julia> pen = Pencil((20, 10, 12), MPI.COMM_WORLD);

julia> u = PencilArray{Float64}(undef, pen);

julia> similar(u) |> summary
"20×10×12 PencilArray{Float64, 3}(::Pencil{3, 2, NoPermutation, Array})"

julia> similar(u, ComplexF32) |> summary
"20×10×12 PencilArray{ComplexF32, 3}(::Pencil{3, 2, NoPermutation, Array})"

julia> similar(u, ComplexF32; singleton = (1, 2)) |> summary
"1×1×12 PencilArray{ComplexF32, 3}(::Pencil{3, 2, NoPermutation, Array})"

julia> similar(u, (4, 3, 8)) |> summary
"4×3×8 Array{Float64, 3}"

julia> similar(u, (4, 3)) |> summary
"4×3 Matrix{Float64}"

julia> similar(u, ComplexF32) |> summary
"20×10×12 PencilArray{ComplexF32, 3}(::Pencil{3, 2, NoPermutation, Array})"

julia> similar(u, ComplexF32, (4, 3)) |> summary
"4×3 Matrix{ComplexF32}"
```

---

    similar(x::PencilArray, [element_type = eltype(x)], p::Pencil; [singleton = ()])

Create a `PencilArray` with the decomposition described by the given `Pencil`.

This variant may be used to create a `PencilArray` that has a different
decomposition than the input `PencilArray`.

# Examples

```jldoctest
julia> pen_u = Pencil((20, 10, 12), (2, 3), MPI.COMM_WORLD);

julia> u = PencilArray{Float64}(undef, pen_u);

julia> pen_v = Pencil(pen_u; decomp_dims = (1, 3), permute = Permutation(2, 3, 1))
Decomposition of 3D data
    Data dimensions: (20, 10, 12)
    Decomposed dimensions: (1, 3)
    Data permutation: Permutation(2, 3, 1)
    Array type: Array

julia> v = similar(u, pen_v);

julia> summary(v)
"20×10×12 PencilArray{Float64, 3}(::Pencil{3, 2, Permutation{(2, 3, 1), 3}, Array})"

julia> pencil(v) === pen_v
true

julia> vint = similar(u, Int, pen_v);

julia> summary(vint)
"20×10×12 PencilArray{Int64, 3}(::Pencil{3, 2, Permutation{(2, 3, 1), 3}, Array})"

julia> pencil(vint) === pen_v
true

julia> similar(u, Int, pen_v; singleton = 1) |> summary
"1×10×12 PencilArray{Int64, 3}(::Pencil{3, 2, Permutation{(2, 3, 1), 3}, Array})"

```
"""
function Base.similar(x::PencilArray, ::Type{S}; singleton = ()) where {S}
    similar(x, S, pencil(x); singleton)
end

# If `dims` is passed, return a regular (non-distributed) array.
Base.similar(x::PencilArray, ::Type{S}, dims::Dims) where {S} =
    similar(parent(x), S, dims)

# Create PencilArray with possibly different Pencil configuration
function Base.similar(x::PencilArray, ::Type{S}, p::Pencil; singleton = ()) where {S}
    dims_loc = _apply_singleton(size_local(p, LogicalOrder()), singleton)
    dims_glb = _apply_singleton(size_global(p, LogicalOrder()), singleton)
    dims_mem = permutation(p) * dims_loc
    dims_ext = extra_dims(x)
    dims_data = (dims_mem..., dims_ext...)
    data = similar(parent(x), S, dims_data)
    PencilArray(p, data, dims_glb)
end

Base.similar(x::PencilArray, p::Pencil; kws...) = similar(x, eltype(x), p; kws...)

# Use same index style as the parent array.
Base.IndexStyle(::Type{<:PencilArray{T,N,A}} where {T,N}) where {A} =
    IndexStyle(A)

# Overload Base._sub2ind for converting from Cartesian to linear index.
@inline function Base._sub2ind(x::PencilArray, I...)
    # _sub2ind(axes(x), I...)  <- default implementation for AbstractArray
    J = permutation(x) * I
    Base._sub2ind(parent(x), J...)
end

# Linear indexing
@propagate_inbounds function Base.getindex(x::PencilArray, i::Integer)
    parent(x)[i]
end

@propagate_inbounds function Base.setindex!(x::PencilArray, v, i::Integer)
    parent(x)[i] = v
end

# Cartesian indexing: assume input indices are unpermuted, and permute them.
# (This is similar to the implementation of PermutedDimsArray.)
@propagate_inbounds Base.getindex(
        x::PencilArray{T,N}, I::Vararg{Int,N}) where {T,N} =
    parent(x)[_genperm(x, I)...]

@propagate_inbounds @inline Base.setindex!(
        x::PencilArray{T,N}, v, I::Vararg{Int,N}) where {T,N} =
    parent(x)[_genperm(x, I)...] = v

@inline function _genperm(x::PencilArray{T,N}, I::NTuple{N,Int}) where {T,N}
    permutation(x) * I
end

@inline _genperm(x::PencilArray, I::CartesianIndex) =
    CartesianIndex(_genperm(x, Tuple(I)))

"""
    pencil(x::PencilArray)

Return decomposition configuration associated to a `PencilArray`.
"""
pencil(x::PencilArray) = x.pencil
pencil(x::PencilArrayCollection) = _only(pencil, x)

"""
    parent(x::PencilArray)

Return array wrapped by a `PencilArray`.
"""
Base.parent(x::PencilArray) = x.data

# This enables aliasing detection (e.g. using Base.mightalias) on PencilArrays.
Base.dataids(x::PencilArray) = Base.dataids(parent(x))

# This is based on strides(::PermutedDimsArray)
function Base.strides(x::PencilArray)
    s = strides(parent(x))
    permutation(x) * s
end

"""
    pointer(x::PencilArray)

Return pointer to the start of the underlying data.

Use with caution: this may not make a lot of sense if the underlying data is not
contiguous or strided (e.g. if the `PencilArray` is wrapping a non-strided
`SubArray`).
"""
Base.pointer(x::PencilArray) = pointer(parent(x))

"""
    ndims_extra(::Type{<:PencilArray})
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
ndims_extra(::Type{<:PencilArray{T,N,A,M,E}}) where {T,N,A,M,E} = E

"""
    ndims_space(x::PencilArray)
    ndims_space(x::PencilArrayCollection)

Number of dimensions associated to the domain geometry.

These dimensions correspond to the leftmost indices of the array.

The total number of dimensions of a `PencilArray` is given by:

    ndims(x) == ndims_space(x) + ndims_extra(x)

"""
ndims_space(x::PencilArray) = ndims(x) - ndims_extra(x)
ndims_space(x::PencilArrayCollection) = _only(ndims_space, x)

"""
    extra_dims(x::PencilArray)
    extra_dims(x::PencilArrayCollection)

Return tuple with size of "extra" dimensions of `PencilArray`.
"""
extra_dims(x::PencilArray) = x.dims_extra
extra_dims(x::PencilArrayCollection) = _only(extra_dims, x)

"""
    sizeof_global(x::PencilArray)
    sizeof_global(x::PencilArrayCollection)

Global size of array in bytes.
"""
sizeof_global(x::PencilArray) = prod(size_global(x)) * sizeof(eltype(x))
sizeof_global(x::PencilArrayCollection) = sum(sizeof_global, x)

"""
    range_local(x::PencilArray, [order = LogicalOrder()])
    range_local(x::PencilArrayCollection, [order = LogicalOrder()])

Local data range held by the `PencilArray`.

By default the dimensions are returned in logical order.
"""
function range_local(x::PencilArray, args...; kws...)
    # TODO should we consider singleton dimensions here?
    range_phys = range_local(pencil(x), args...; kws...)
    N = length(range_phys)
    range_extr = ntuple(i -> axes(x, N + i), Val(ndims_extra(x)))
    (range_phys..., range_extr...)
end

range_local(x::PencilArrayCollection, args...) = _only(range_local, x, args...)

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
    (range_remote(pencil(x), args...)..., map(Base.OneTo, extra_dims(x))...)

"""
    get_comm(x::PencilArray)
    get_comm(x::PencilArrayCollection)

Get MPI communicator associated to a pencil-distributed array.
"""
get_comm(x::MaybePencilArrayCollection) = get_comm(pencil(x))

"""
    permutation(::Type{<:PencilArray})
    permutation(x::PencilArray)
    permutation(x::PencilArrayCollection)

Get index permutation associated to the given `PencilArray`.

Returns `NoPermutation()` if there is no associated permutation.
"""
function permutation end

function permutation(::Type{A}) where {A <: PencilArray}
    P = pencil_type(A)
    perm = permutation(P)
    E = ndims_extra(A)
    append(perm, Val(E))
end

permutation(x::PencilArray) = permutation(typeof(x))
permutation(x::PencilArrayCollection) = _only(permutation, x)

"""
    topology(x::PencilArray)
    topology(x::PencilArrayCollection)

Get [`MPITopology`](@ref) associated to a `PencilArray`.
"""
topology(x::MaybePencilArrayCollection) = topology(pencil(x))

## Common array operations
# We try to avoid falling onto the generic AbstractArray interface, because it
# generally uses scalar indexing which is not liked by GPU arrays.
Base.zero(x::PencilArray) = fill!(similar(x), zero(eltype(x)))

function _check_compatible_arrays(x::PencilArray, y::PencilArray)
    # The condition is stronger than needed, but it's pretty common for arrays
    # to share the same Pencil, and it's more efficient to compare this way.
    pencil(x) === pencil(y) ||
        throw(ArgumentError("arrays are not compatible"))
end

function Base.copyto!(x::PencilArray, y::PencilArray)
    _check_compatible_arrays(x, y)
    copyto!(parent(x), parent(y))
    x
end

# Should this be an equality across all MPI processes?
function Base.:(==)(x::PencilArray, y::PencilArray)
    _check_compatible_arrays(x, y)
    parent(x) == parent(y)
end

function Base.isapprox(x::PencilArray, y::PencilArray; kws...)
    _check_compatible_arrays(x, y)
    isapprox(parent(x), parent(y); kws...)
end

function Base.fill!(A::PencilArray, x)
    fill!(parent(A), x)
    A
end

"""
    typeof_ptr(x::AbstractArray)
    typeof_ptr(x::PencilArray)

Get the type of pointer to the underlying array of a `PencilArray` or `AbstractArray`.
"""
typeof_ptr(A::AbstractArray) = typeof(pointer(A)).name.wrapper

"""
    typeof_array(x::Pencil)
    typeof_array(x::PencilArray)
    typeof_array(x::AbstractArray)

Get the type of array (without the element type) so it can be used as a constructor.
"""
typeof_array(A::PencilArray) = typeof_array(parent(A))

"""
    localgrid(x::PencilArray, args...)

Equivalent of `localgrid(pencil(x), args...)`.
"""
LocalGrids.localgrid(A::PencilArray, args...) = localgrid(pencil(A), args...)
