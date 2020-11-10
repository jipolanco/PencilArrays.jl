import ArrayInterface:
    ArrayInterface,
    parent_type,
    Contiguous,
    contiguous_axis,
    contiguous_batch_size,
    StrideRank, stride_rank,
    DenseDims, dense_dims

parent_type(::Type{<:PencilArray{T,N,A}}) where {T,N,A} = A

contiguous_axis(::Type{A}) where {A <: PencilArray} =
    _contiguous_axis(
        contiguous_axis(parent_type(A)),
        get_permutation(A),
    )

_contiguous_axis(x::Nothing, ::Any) = x
_contiguous_axis(x::Contiguous, ::NoPermutation) = x
@inline function _contiguous_axis(x::Contiguous{i}, p::Permutation) where {i}
    i == -1 && return x
    Contiguous(Tuple(p)[i])
end

contiguous_batch_size(::Type{A}) where {A <: PencilArray} =
    contiguous_batch_size(parent_type(A))

stride_rank(::Type{A}) where {A <: PencilArray} =
    _stride_rank(
        stride_rank(parent_type(A)),
        inverse_permutation(get_permutation(A)),
    )

_stride_rank(x::Nothing, iperm) = x
_stride_rank(x::StrideRank, ::NoPermutation) = x
@inline _stride_rank(::StrideRank{R}, iperm::Permutation) where {R} =
    StrideRank(Tuple(iperm))

dense_dims(::Type{A}) where {A <: PencilArray} =
    _dense_dims(
        dense_dims(parent_type(A)),
        inverse_permutation(get_permutation(A)),
    )

_dense_dims(x::Nothing, perm) = x
_dense_dims(x::DenseDims, ::NoPermutation) = x
@inline _dense_dims(x::DenseDims, perm::Permutation) = x[Val(Tuple(perm))]

ArrayInterface.size(A::PencilArray) =
    permute_indices(ArrayInterface.size(parent(A)), get_permutation(A))

ArrayInterface.strides(A::PencilArray) =
    permute_indices(ArrayInterface.strides(parent(A)), get_permutation(A))
