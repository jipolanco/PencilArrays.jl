import ArrayInterface:
    ArrayInterface,
    StaticInt,
    contiguous_axis,
    contiguous_batch_size,
    parent_type,
    stride_rank,
    dense_dims

parent_type(::Type{<:PencilArray{T,N,A}}) where {T,N,A} = A

contiguous_axis(::Type{A}) where {A <: PencilArray} =
    _contiguous_axis(
        contiguous_axis(parent_type(A)),
        permutation(A),
    )

_contiguous_axis(x::Nothing, ::AbstractPermutation) = x
_contiguous_axis(x::StaticInt, ::NoPermutation) = x
@inline function _contiguous_axis(x::StaticInt{i}, p::Permutation) where {i}
    i == -1 && return x
    StaticInt(p[Val(i)])
end

contiguous_batch_size(::Type{A}) where {A <: PencilArray} =
    contiguous_batch_size(parent_type(A))

function stride_rank(::Type{A}) where {A <: PencilArray}
    rank = stride_rank(parent_type(A))
    rank === nothing && return nothing
    iperm = Tuple(inv(permutation(A)))
    iperm === nothing && return rank
    ArrayInterface.permute(rank, Val(iperm))
end

function dense_dims(::Type{A}) where {A <: PencilArray}
    dense = dense_dims(parent_type(A))
    dense === nothing && return nothing
    perm = Tuple(inv(permutation(A)))
    perm === nothing && return dense
    ArrayInterface.permute(dense, Val(perm))
end

ArrayInterface.size(A::PencilArray) =
    permutation(A) * ArrayInterface.size(parent(A))

ArrayInterface.strides(A::PencilArray) =
    permutation(A) * ArrayInterface.strides(parent(A))
