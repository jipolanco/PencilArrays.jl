# We make LinearIndices(::PencilArray) return a PermutedLinearIndices, which
# takes index permutation into account.
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
Base.CartesianIndices(A::PencilArray) =
    PermutedCartesianIndices(CartesianIndices(parent(A)), permutation(A))

function Base.CartesianIndices(g::GlobalPencilArray)
    p = permutation(g)
    axs_log = axes(g)      # offset axes in logical (unpermuted) order
    axs_mem = p * axs_log  # offset axes in memory (permuted) order
    PermutedCartesianIndices(CartesianIndices(axs_mem), p)
end
