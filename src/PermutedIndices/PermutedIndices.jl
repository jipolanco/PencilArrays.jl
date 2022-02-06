module PermutedIndices

import ..Permutations: permutation
using StaticPermutations

export PermutedLinearIndices, PermutedCartesianIndices

# Custom definitions of LinearIndices and CartesianIndices to take into account
# index permutations.
#
# In particular, when array dimensions are permuted, the default
# CartesianIndices do not iterate in memory order, making them suboptimal.
# We try to workaround that by adding a custom definition of CartesianIndices.
#
# (TODO Better / cleaner way to do this??)

struct PermutedLinearIndices{
        N, L <: LinearIndices, Perm,
    } <: AbstractArray{Int,N}
    data :: L  # indices in permuted order
    perm :: Perm
    function PermutedLinearIndices(
            ind::LinearIndices{N}, perm::Perm) where {N, Perm}
        L = typeof(ind)
        new{N, L, Perm}(ind, perm)
    end
end

permutation(L::PermutedLinearIndices) = L.perm

Base.length(L::PermutedLinearIndices) = length(L.data)
Base.size(L::PermutedLinearIndices) = permutation(L) \ size(L.data)
Base.axes(L::PermutedLinearIndices) = permutation(L) \ axes(L.data)
Base.iterate(L::PermutedLinearIndices, args...) = iterate(L.data, args...)
Base.lastindex(L::PermutedLinearIndices) = lastindex(L.data)

@inline function Base.getindex(L::PermutedLinearIndices, i::Integer)
    @boundscheck checkbounds(L.data, i)
    @inbounds L.data[i]
end

# Input: indices in logical (unpermuted) order
@inline function Base.getindex(
        L::PermutedLinearIndices{N}, I::Vararg{Integer,N},
    ) where {N}
    J = permutation(L) * I
    @boundscheck checkbounds(L.data, J...)
    @inbounds L.data[J...]
end

struct PermutedCartesianIndices{
        N, C <: CartesianIndices{N}, Perm,
    } <: AbstractArray{CartesianIndex{N}, N}
    data :: C     # indices in memory (permuted) order
    perm :: Perm  # permutation (logical -> memory)
    function PermutedCartesianIndices(ind::CartesianIndices{N},
                                      perm::Perm) where {N, Perm}
        C = typeof(ind)
        new{N, C, Perm}(ind, perm)
    end
end

permutation(C::PermutedCartesianIndices) = C.perm

Base.size(C::PermutedCartesianIndices) = permutation(C) \ size(C.data)
Base.axes(C::PermutedCartesianIndices) = permutation(C) \ axes(C.data)

@inline function Base.iterate(C::PermutedCartesianIndices, args...)
    next = iterate(C.data, args...)
    next === nothing && return nothing
    I, state = next  # `I` has permuted indices
    J = permutation(C) \ I   # unpermute indices
    J, state
end

# Get i-th Cartesian index in memory (permuted) order.
# Returns the Cartesian index in logical (unpermuted) order.
@inline function Base.getindex(C::PermutedCartesianIndices, i::Integer)
    @boundscheck checkbounds(C.data, i)
    @inbounds I = C.data[i]  # convert linear to Cartesian index (relatively slow...)
    permutation(C) \ I       # unpermute indices
end

# Not sure if this is what makes the most sense, but it's consistent with the
# behaviour of CartesianIndices(::OffsetArray). In any case, this function is
# mostly used for printing (it's used by show(::PermutedCartesianIndices)), and
# almost never for actual computations.
@inline function Base.getindex(
        C::PermutedCartesianIndices{N}, I::Vararg{Integer,N},
    ) where {N}
    @boundscheck checkbounds(C, I...)
    CartesianIndex(I)
end

end
