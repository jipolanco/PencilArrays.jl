module LocalGrids

using ..Pencils
using StaticPermutations

export
    localgrid

"""
    AbstractLocalGrid{N, P <: AbstractPermutation}

Abstract type specifying the local portion of an `N`-dimensional grid.
"""
abstract type AbstractLocalGrid{N, P <: AbstractPermutation} end

Base.ndims(::Type{<:AbstractLocalGrid{N}}) where {N} = N
Base.ndims(g::AbstractLocalGrid) = ndims(typeof(g))
Pencils.permutation(g::AbstractLocalGrid) = g.perm
coordinates(g::AbstractLocalGrid) = g.coords

"""
    LocalRectilinearGrid{N, P} <: AbstractLocalGrid{N, P}

Defines the local portion of a rectilinear grid in `N` dimensions.

A rectilinear grid is represented by a set of orthogonal coordinates `(x, y, z, ...)`.
"""
struct LocalRectilinearGrid{
        N,
        P,
        LocalCoords <: Tuple{Vararg{AbstractVector, N}},
    } <: AbstractLocalGrid{N, P}
    perm   :: P
    coords :: LocalCoords  # in logical order
end

function LocalRectilinearGrid(
        p::Pencil{N}, coords_global::Tuple{Vararg{AbstractVector, N}},
    ) where {N}
    ranges = range_local(p, LogicalOrder())
    coords_local = map(view, coords_global, ranges)
    perm = permutation(p)
    LocalRectilinearGrid(perm, coords_local)
end

function Base.show(io::IO, g::LocalRectilinearGrid{N, P}) where {N, P}
    print(io, nameof(typeof(g)), "{$N, $P} with coordinates:")
    map(enumerate(coordinates(g))) do (n, xs)
        print(io, "\n ($n) $xs")
    end
end

# For convenience when working with up to three dimensions.
@inline function Base.getproperty(g::LocalRectilinearGrid, name::Symbol)
    if ndims(g) ≥ 1 && name === :x
        coordinates(g)[1]
    elseif ndims(g) ≥ 2 && name === :y
        coordinates(g)[2]
    elseif ndims(g) ≥ 3 && name === :z
        coordinates(g)[3]
    else
        getfield(g, name)
    end
end

"""
    localgrid(p::Pencil, (x_global, y_global, ...)) -> LocalRectilinearGrid

Create a [`LocalRectilinearGrid`](@ref) from a decomposition configuration and
from a set of orthogonal global coordinates `(x_global, y_global, ...)`.

In this case, each `*_global` is an `AbstractVector` describing the coordinates
along one dimension of the global grid.
"""
localgrid(p::Pencil, coords_global::Tuple{Vararg{AbstractVector}}) =
    LocalRectilinearGrid(p, coords_global)

end
