module LocalGrids

import ..Permutations: permutation
using ..PermutedIndices

using Base.Broadcast
using Base: @propagate_inbounds
using StaticPermutations

export localgrid

"""
    AbstractLocalGrid{N, Perm <: AbstractPermutation}

Abstract type specifying the local portion of an `N`-dimensional grid.
"""
abstract type AbstractLocalGrid{N, Perm <: AbstractPermutation} end

Base.ndims(::Type{<:AbstractLocalGrid{N}}) where {N} = N
Base.ndims(g::AbstractLocalGrid) = ndims(typeof(g))
permutation(g::AbstractLocalGrid) = getfield(g, :perm)

"""
    LocalGrids.components(g::LocalRectilinearGrid) -> (xs, ys, zs, ...)

Get coordinates associated to the current MPI process.
"""
components(g::AbstractLocalGrid) = getfield(g, :coords)

include("rectilinear.jl")

end
