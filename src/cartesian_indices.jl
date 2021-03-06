# Custom definitions of LinearIndices and CartesianIndices to take into account
# index permutations.
#
# In particular, when array dimensions are permuted, the default
# CartesianIndices do not iterate in memory order, making them suboptimal.
# We try to workaround that by adding a custom definition of CartesianIndices.
#
# (TODO Better / cleaner way to do this??)

# We make LinearIndices(::PencilArray) return a PermutedLinearIndices, which
# takes index permutation into account.
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

Base.length(L::PermutedLinearIndices) = length(L.data)
Base.size(L::PermutedLinearIndices) = L.perm \ size(L.data)
Base.axes(L::PermutedLinearIndices) = L.perm \ axes(L.data)
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
    J = L.perm * I
    @boundscheck checkbounds(L.data, J...)
    @inbounds L.data[J...]
end

Base.LinearIndices(A::PencilArray) =
    PermutedLinearIndices(LinearIndices(parent(A)), permutation(A))

function Base.LinearIndices(g::GlobalPencilArray)
    p = permutation(g)
    axs_log = axes(g)      # offset axes in logical (unpermuted) order
    axs_mem = p * axs_log  # offset axes in memory (permuted) order
    PermutedLinearIndices(LinearIndices(axs_mem), p)
end

# We make CartesianIndices(::PencilArray) return a PermutedCartesianIndices,
# which loops faster (in memory order) when there are index permutations.
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

Base.size(C::PermutedCartesianIndices) = C.perm \ size(C.data)
Base.axes(C::PermutedCartesianIndices) = C.perm \ axes(C.data)

@inline function Base.iterate(C::PermutedCartesianIndices, args...)
    next = iterate(C.data, args...)
    next === nothing && return nothing
    I, state = next  # `I` has permuted indices
    J = C.perm \ I   # unpermute indices
    J, state
end

# Get i-th Cartesian index in memory (permuted) order.
# Returns the Cartesian index in logical (unpermuted) order.
@inline function Base.getindex(C::PermutedCartesianIndices, i::Integer)
    @boundscheck checkbounds(C.data, i)
    @inbounds I = C.data[i]  # convert linear to Cartesian index (relatively slow...)
    C.perm \ I           # unpermute indices
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

Base.CartesianIndices(A::PencilArray) =
    PermutedCartesianIndices(CartesianIndices(parent(A)), permutation(A))

function Base.CartesianIndices(g::GlobalPencilArray)
    p = permutation(g)
    axs_log = axes(g)      # offset axes in logical (unpermuted) order
    axs_mem = p * axs_log  # offset axes in memory (permuted) order
    PermutedCartesianIndices(CartesianIndices(axs_mem), p)
end
