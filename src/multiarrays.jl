## Definition of AbstractManyPencilArray

"""
    AbstractManyPencilArray{N,M}

Abstract type specifying a container holding `M` different
[`PencilArray`](@ref) views to the same underlying.

All views share the same dimensionality `N`. In principle, their element types
can be different.

A concrete implementation, [`ManyPencilArray`](@ref), is proposed in which all
arrays have the same element type `T`.
"""
abstract type AbstractManyPencilArray{N,M} end

Base.ndims(::AbstractManyPencilArray{N}) where {N} = N

"""
    length(A::AbstractManyPencilArray)

Returns the number of [`PencilArray`](@ref)s wrapped by `A`.
"""
Base.length(::AbstractManyPencilArray{N,M}) where {N,M} = M

"""
    Tuple(A::AbstractManyPencilArray) -> (u1, u2, …)

Returns the [`PencilArray`](@ref)s wrapped by `A`.

This can be useful for iterating over all the wrapped arrays.
"""
@inline Base.Tuple(A::AbstractManyPencilArray) = A.arrays

"""
    first(A::AbstractManyPencilArray)

Returns the first [`PencilArray`](@ref) wrapped by `A`.
"""
Base.first(A::AbstractManyPencilArray) = first(A.arrays)

"""
    last(A::AbstractManyPencilArray)

Returns the last [`PencilArray`](@ref) wrapped by `A`.
"""
Base.last(A::AbstractManyPencilArray) = last(A.arrays)

"""
    getindex(A::AbstractManyPencilArray, ::Val{i})
    getindex(A::AbstractManyPencilArray, i::Integer)

Returns the i-th [`PencilArray`](@ref) wrapped by `A`.

If possible, the `Val{i}` form should be preferred, as it ensures that the full
type of the returned `PencilArray` is known by the compiler.

See also [`first(::AbstractManyPencilArray)`](@ref), [`last(::AbstractManyPencilArray)`](@ref).

# Example

```julia
A = ManyPencilArray(pencil1, pencil2, pencil3)

# Get the PencilArray associated to `pencil2`.
u2 = A[2]
u2 = A[Val(2)]
```
"""
Base.getindex(A::AbstractManyPencilArray, ::Val{i}) where {i} =
    _getindex(Val(i), A.arrays...)
@inline Base.getindex(A::AbstractManyPencilArray, i) = A[Val(i)]

@inline function _getindex(::Val{i}, a, t::Vararg) where {i}
    i :: Integer
    i <= 0 && throw(BoundsError("index must be >= 1"))
    i == 1 && return a
    _getindex(Val(i - 1), t...)
end

# This will happen if the index `i` intially passed is too large.
@inline _getindex(::Val) = throw(BoundsError("invalid index"))

# ====================================================================== #
## Concrete implementation: ManyPencilArray

"""
    ManyPencilArray{T,N,M} <: AbstractManyPencilArray{N,M}

Container holding `M` different [`PencilArray`](@ref) views to the same
underlying data buffer. All views share the same element type `T` and
dimensionality `N`.

This can be used to perform in-place data transpositions with
[`transpose!`](@ref Transpositions.transpose!).

---

    ManyPencilArray{T}(undef, pencils...; extra_dims=())

Create a `ManyPencilArray` container that can hold data of type `T` associated
to all the given [`Pencil`](@ref)s.

The optional `extra_dims` argument is the same as for [`PencilArray`](@ref).
"""
struct ManyPencilArray{
        T,  # element type of each array
        N,  # number of dimensions of each array (including extra_dims)
        M,  # number of arrays
        Arrays <: Tuple{Vararg{PencilArray,M}},
        DataVector <: AbstractVector{T},
    } <: AbstractManyPencilArray{N, M}
    data   :: DataVector
    arrays :: Arrays

    function ManyPencilArray{T}(
            init, pfirst::Pencil{Np}, pens::Vararg{Pencil{Np}};
            extra_dims::Dims=(),
        ) where {Np,T}
        pencils = (pfirst, pens...)
        BufType = typeof_array(pfirst)
        @assert all(p -> typeof_array(p) === BufType, pens)
        data_length = max(length.(pencils)...) * prod(extra_dims)
        data = BufType{T}(init, data_length)
        arrays = _make_arrays(data, extra_dims, pencils...)
        N = Np + length(extra_dims)
        M = length(pencils)
        new{T, N, M, typeof(arrays), typeof(data)}(data, arrays)
    end
end

function _make_arrays(data::DenseVector{T}, extra_dims::Dims, p::Pencil,
                      pens::Vararg{Pencil}) where {T}
    dims = (size_local(p, MemoryOrder())..., extra_dims...)
    n = prod(dims)
    @assert n == length_local(p) * prod(extra_dims)
    vec = unsafe_wrap(typeof(data), pointer(data), n) # fixes efficiency issues with vec = view(data, Base.OneTo(n))
    arr = reshape(vec, dims)
    A = PencilArray(p, arr)
    (A, _make_arrays(data, extra_dims, pens...)...)
end

_make_arrays(::DenseVector, ::Dims) = ()

Base.eltype(::ManyPencilArray{T}) where {T} = T
