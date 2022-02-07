module LocalGrids

import ..Permutations: permutation
using ..PermutedIndices

using Base.Broadcast
using StaticPermutations

export localgrid

"""
    AbstractLocalGrid{N, Perm <: AbstractPermutation}

Abstract type specifying the local portion of an `N`-dimensional grid.
"""
abstract type AbstractLocalGrid{N, Perm <: AbstractPermutation} end

Base.ndims(::Type{<:AbstractLocalGrid{N}}) where {N} = N
Base.ndims(g::AbstractLocalGrid) = ndims(typeof(g))
permutation(g::AbstractLocalGrid) = g.perm

"""
    LocalGrids.components(g::LocalRectilinearGrid) -> (xs, ys, zs, ...)

Get coordinates associated to the current MPI process.
"""
components(g::AbstractLocalGrid) = g.coords

include("rectilinear.jl")

end
