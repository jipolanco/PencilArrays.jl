import ArrayInterface:
    parent_type,
    contiguous_axis,
    Contiguous

parent_type(::Type{<:PencilArray{T,N,A}}) where {T,N,A} = A

contiguous_axis(::Type{A}) where {A <: PencilArray} =
    _contiguous_axis(
        contiguous_axis(parent_type(A)),
        get_permutation(A),
    )

_contiguous_axis(x::Nothing, perm) = x
_contiguous_axis(x::Contiguous{-1}, perm) = x
_contiguous_axis(x::Contiguous, ::NoPermutation) = x
@inline _contiguous_axis(::Contiguous{i}, p::Permutation) where {i} =
    Contiguous(Tuple(p)[i])
