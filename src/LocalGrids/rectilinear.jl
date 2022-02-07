"""
    LocalRectilinearGrid{N, Perm} <: AbstractLocalGrid{N, Perm}

Defines the local portion of a rectilinear grid in `N` dimensions.

A rectilinear grid is represented by a set of orthogonal coordinates `(x, y, z, ...)`.
"""
struct LocalRectilinearGrid{
        N,
        Perm <: AbstractPermutation,
        LocalCoords <: Tuple{Vararg{AbstractVector, N}},
    } <: AbstractLocalGrid{N, Perm}
    coords :: LocalCoords  # in logical order
    perm   :: Perm
end

"""
    localgrid((xs, ys, ...), perm = NoPermutation()) -> LocalRectilinearGrid

Create a [`LocalRectilinearGrid`](@ref) from a set of orthogonal coordinates
`(xs, ys, ...)`, where each element is an `AbstractVector`.

Optionally, one can pass a static permutation (as in `Permutation(2, 1, 3)`) to
change the order in which the coordinates are iterated.
"""
function localgrid(
        coords::Tuple{Vararg{AbstractVector}},
        perm::AbstractPermutation = NoPermutation(),
    )
    LocalRectilinearGrid(coords, perm)
end

# Axes in logical order
Base.axes(g::LocalRectilinearGrid) = map(xs -> axes(xs, 1), components(g))

# These are needed for `collect`
Base.length(g::LocalRectilinearGrid) = prod(xs -> length(xs), components(g))

@generated function Base.eltype(
        ::Type{<:LocalRectilinearGrid{N, P, VecTuple}}
    ) where {N, P, VecTuple}
    types = Tuple{map(eltype, VecTuple.parameters)...}
    :( $types )
end

# We define this wrapper type to be able to control broadcasting on separate
# grid components (x, y, ...).
struct RectilinearGridComponent{
        i,  # dimension of this coordinate
        FullGrid <: LocalRectilinearGrid,  # dataset dimension
        Coords <: AbstractVector,
    }
    grid :: FullGrid
    data :: Coords
    @inline function RectilinearGridComponent(
            g::LocalRectilinearGrid, ::Val{i},
        ) where {i}
        data = components(g)[i]
        new{i, typeof(g), typeof(data)}(g, data)
    end
end

function Base.show(io::IO, xs::RectilinearGridComponent{i}) where {i}
    print(io, "Component i = $i of ")
    summary(io, xs.grid)
    print(io, ": ", xs.data)
    nothing
end

@inline Base.getindex(g::LocalRectilinearGrid, i::Val) =
    RectilinearGridComponent(g, i)
@inline Base.getindex(g::LocalRectilinearGrid, i::Int) = g[Val(i)]

@inline function Base.getindex(
        g::LocalRectilinearGrid{N}, inds::Vararg{Integer, N},
    ) where {N}
    @boundscheck checkbounds(CartesianIndices(axes(g)), inds...)
    map((xs, i) -> @inbounds(xs[i]), components(g), inds)
end

@inline Base.getindex(g::LocalRectilinearGrid, I::CartesianIndex) =
    g[Tuple(I)...]

@inline function Base.CartesianIndices(g::LocalRectilinearGrid)
    perm = permutation(g)
    axs = perm * axes(g)          # axes in memory order
    inds = CartesianIndices(axs)  # each index inds[i] is in memory order
    PermutedCartesianIndices(inds, perm)
end

# This is used by eachindex(::LocalRectilinearGrid)
@inline Base.keys(g::LocalRectilinearGrid) = CartesianIndices(g)

@inline function Base.iterate(g::LocalRectilinearGrid, state = nothing)
    perm = permutation(g)
    stuff = if state === nothing
        # Create and advance actual iterator
        coords_mem = perm * components(g)  # iterate in memory order
        iter = Iterators.product(coords_mem...)
        iterate(iter)
    else
        iter = first(state)
        iterate(state...)
    end
    stuff === nothing && return nothing
    x⃗_mem, next = stuff
    x⃗_log = perm \ x⃗_mem  # current coordinate in logical order (x, y, z, ...)
    x⃗_log, (iter, next)
end

function Broadcast.broadcastable(xs::RectilinearGridComponent{i}) where {i}
    g = xs.grid
    N = ndims(g)
    perm = permutation(g)
    data = xs.data
    dims = ntuple(j -> j == i ? length(data) : 1, Val(N))
    reshape(xs.data, perm * dims)
end

function Base.show(io::IO, g::LocalRectilinearGrid{N}) where {N}
    print(io, nameof(typeof(g)), "{$N} with ")
    perm = permutation(g)
    isidentity(perm) || print(io, perm, " and ")
    print(io, "coordinates:")
    foreach(enumerate(components(g))) do (n, xs)
        print(io, "\n ($n) $xs")
    end
    nothing
end

function Base.summary(io::IO, g::LocalRectilinearGrid)
    N = ndims(g)
    print(io, nameof(typeof(g)), "{$N}")
    nothing
end

# For convenience when working with up to three dimensions.
@inline function Base.getproperty(g::LocalRectilinearGrid, name::Symbol)
    if ndims(g) ≥ 1 && name === :x
        g[Val(1)]
    elseif ndims(g) ≥ 2 && name === :y
        g[Val(2)]
    elseif ndims(g) ≥ 3 && name === :z
        g[Val(3)]
    else
        getfield(g, name)
    end
end
