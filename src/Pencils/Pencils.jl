module Pencils

using StaticPermutations

using MPI
using Reexport
using StaticArrays: SVector
using TimerOutputs

export Pencil, MPITopology
export Permutation, NoPermutation  # from StaticPermutations
export MemoryOrder, LogicalOrder
export decomposition, permutation
export get_comm, timer
export topology
export range_local, range_remote, size_local, size_global, to_local

# Describes the portion of an array held by a given MPI process.
const ArrayRegion{N} = NTuple{N,UnitRange{Int}} where N

# Deprecated in v0.4
@deprecate get_permutation permutation
@deprecate get_decomposition decomposition
@deprecate get_timer timer

include("MPITopologies.jl")
@reexport using .MPITopologies
import .MPITopologies: get_comm

include("data_ranges.jl")
include("index_orders.jl")

"""
    Pencil{N,M}

Describes the decomposition of an `N`-dimensional array among MPI processes
along `M` directions (with `M < N`).

---

    Pencil(
        topology::MPITopology{M}, size_global::Dims{N}, decomp_dims::Dims{M};
        permute::AbstractPermutation = NoPermutation(),
        timer = TimerOutput(),
    )

Define the decomposition of an `N`-dimensional geometry along `M` dimensions.

The dimensions of the geometry are given by `size_global = (N1, N2, ...)`. The
`Pencil` describes the decomposition of an array of dimensions `size_global`
across a group of MPI processes.

Data is distributed over the given `M`-dimensional MPI topology (with `M < N`).
The decomposed dimensions are given by `decomp_dims`.

The optional parameter `perm` should be a (compile-time) tuple defining a
permutation of the data indices. Such permutation may be useful for performance
reasons, since it may be preferable (e.g. for FFTs) that the data is contiguous
along the pencil direction.

It is also possible to pass a `TimerOutput` to the constructor. See
[Measuring performance](@ref PencilArrays.measuring_performance) for details.

# Examples

Decompose a 3D geometry of global dimensions ``N_x × N_y × N_z = 4×8×12`` along
the second (``y``) and third (``z``) dimensions.
```julia
Pencil(topology, (4, 8, 12), (2, 3))                                # data is in (x, y, z) order
Pencil(topology, (4, 8, 12), (2, 3), permute=Permutation(3, 2, 1))  # data is in (z, y, x) order
```
In the second case, the actual data is stored in `(z, y, x)` order within
each MPI process.

---

    Pencil(
        p::Pencil{N,M};
        decomp_dims::Dims{M} = decomposition(p),
        size_global::Dims{N} = size_global(p),
        permute::P = permutation(p),
        timer::TimerOutput = timer(p),
    )

Create new pencil configuration from an existent one.

This constructor enables sharing temporary data buffers between the two pencil
configurations, leading to reduced global memory usage.
"""
struct Pencil{
        N,  # spatial dimensions
        M,  # MPI topology dimensions (< N)
        P,  # optional index permutation (see Permutation)
    }
    # M-dimensional MPI decomposition info (with M < N).
    topology :: MPITopology{M}

    # Global array dimensions (N1, N2, ...).
    # These dimensions are *before* permutation by perm.
    size_global :: Dims{N}

    # Decomposition directions (sorted in increasing order).
    # Example: for x-pencils, this is (2, 3, ..., N).
    decomp_dims :: Dims{M}

    # Part of the array held by every process.
    # These dimensions are *before* permutation by `perm`.
    axes_all :: Array{ArrayRegion{N}, M}

    # Part of the array held by the local process (before permutation).
    axes_local :: ArrayRegion{N}

    # Part of the array held by the local process (after permutation).
    axes_local_perm :: ArrayRegion{N}

    # Optional axes permutation.
    perm :: P

    # Data buffers for transpositions.
    send_buf :: Vector{UInt8}
    recv_buf :: Vector{UInt8}

    # Timing information.
    timer :: TimerOutput

    function Pencil(
            topology::MPITopology{M}, size_global::Dims{N}, decomp_dims::Dims{M};
            permute::AbstractPermutation = NoPermutation(),
            send_buf = UInt8[], recv_buf = UInt8[],
            timer = TimerOutput(),
        ) where {N, M}
        check_permutation(permute)
        _check_selected_dimensions(N, decomp_dims)
        decomp_dims = _sort_dimensions(decomp_dims)
        axes_all = get_axes_matrix(decomp_dims, topology.dims, size_global)
        axes_local = axes_all[coords_local(topology)...]
        axes_local_perm = permute * axes_local
        P = typeof(permute)
        new{N,M,P}(topology, size_global, decomp_dims, axes_all, axes_local,
                   axes_local_perm, permute, send_buf, recv_buf, timer)
    end

    function Pencil(p::Pencil{N,M};
                    decomp_dims::Dims{M} = decomposition(p),
                    size_global::Dims{N} = size_global(p),
                    permute = permutation(p),
                    timer::TimerOutput = timer(p),
                    etc...,
                   ) where {N, M}
        Pencil(p.topology, size_global, decomp_dims;
               permute=permute, timer=timer,
               send_buf=p.send_buf, recv_buf=p.recv_buf,
               etc...)
    end
end

function check_permutation(perm)
    isperm(perm) && return
    throw(ArgumentError("invalid permutation of dimensions: $perm"))
end

# Verify that `dims` is a subselection of dimensions in 1:N.
function _check_selected_dimensions(N, dims::Dims{M}) where M
    if M >= N
        throw(ArgumentError(
            "number of decomposed dimensions `M` must be less than the " *
            "total number of dimensions N = $N (got M = $M)"))
    end
    if !allunique(dims)
        throw(ArgumentError("dimensions may not be repeated. Got $dims."))
    end
    if !all(1 .<= dims .<= N)
        throw(ArgumentError("dimensions must be in 1:$N. Got $dims."))
    end
    nothing
end

# Use the `sort` method from StaticArrays and convert back to tuple.
_sort_dimensions(dims::Dims{N}) where {N} = Tuple(sort(SVector(dims)))

function Base.show(io::IO, p::Pencil)
    perm = permutation(p)
    print(io,
          """
          Decomposition of $(ndims(p))D data
              Data dimensions: $(size_global(p))
              Decomposed dimensions: $(decomposition(p))
              Data permutation: $(perm)""")
end

"""
    timer(p::Pencil)

Get `TimerOutput` attached to a `Pencil`.

See [Measuring performance](@ref PencilArrays.measuring_performance) for details.
"""
timer(p::Pencil) = p.timer

"""
    ndims(p::Pencil)

Number of spatial dimensions associated to pencil data.

This corresponds to the total number of dimensions of the space, which includes
the decomposed and non-decomposed dimensions.
"""
Base.ndims(::Pencil{N}) where N = N

"""
    get_comm(p::Pencil)

Get MPI communicator associated to an MPI decomposition scheme.
"""
get_comm(p::Pencil) = get_comm(p.topology)

"""
    permutation(::Type{<:Pencil}) -> AbstractPermutation
    permutation(p::Pencil)        -> AbstractPermutation

Get index permutation associated to the given pencil configuration.

Returns `NoPermutation()` if there is no associated permutation.
"""
permutation(p::Pencil) = permutation(typeof(p))
permutation(::Type{<:Pencil{N,M,P}}) where {N,M,P} = _instanceof(P)

@inline _instanceof(::Type{T}) where {T <: AbstractPermutation} = T()
@inline _instanceof(::Type{<:Permutation{p}}) where {p} = Permutation(p)

"""
    decomposition(p::Pencil)

Get tuple with decomposed dimensions of the given pencil configuration.
"""
decomposition(p::Pencil) = p.decomp_dims

"""
    topology(p::Pencil)

Get [`MPITopology`](@ref) attached to `Pencil`.
"""
topology(p::Pencil) = p.topology

"""
    length(p::Pencil)

Get linear length of data associated to the local pencil layout.
"""
Base.length(p::Pencil) = prod(size_local(p))

"""
    range_local(p::Pencil, [order = LogicalOrder()])

Local data range held by the pencil.

By default the dimensions are not permuted, i.e. they follow the logical order
of dimensions.
"""
range_local(p::Pencil, ::LogicalOrder) = p.axes_local
range_local(p::Pencil, ::MemoryOrder) = p.axes_local_perm
range_local(p) = range_local(p, DefaultOrder())

"""
    range_remote(p::Pencil, coords, [order = LogicalOrder()])
    range_remote(p::Pencil, n::Integer, [order = LogicalOrder()])

Get data range held by a given MPI process.

In the first variant, `coords` are the coordinates of the MPI process in
the Cartesian topology. They can be specified as a tuple `(i, j, ...)` or as a
`CartesianIndex`.

In the second variant, `n` is the linear index of a given process in the
topology.
"""
range_remote(p::Pencil, n::Integer, ::LogicalOrder) = p.axes_all[n]
range_remote(p::Pencil{N,M}, I::CartesianIndex{M}, ::LogicalOrder) where {N,M} =
    p.axes_all[I]
range_remote(p::Pencil{N,M}, I::Dims{M}, ::LogicalOrder) where {N,M} =
    range_remote(p, CartesianIndex(I), LogicalOrder())
range_remote(p, I) = range_remote(p, I, LogicalOrder())
range_remote(p, I, ::MemoryOrder) =
    permutation(p) * range_remote(p, I, LogicalOrder())

"""
    size_local(p::Pencil, [order = LogicalOrder()])

Local dimensions of the data held by the pencil.

By default the dimensions are not permuted, i.e. they follow the logical order
of dimensions.
"""
size_local(p::Pencil, etc...) = map(length, range_local(p, etc...))

"""
    size_global(p::Pencil, [order = LogicalOrder()])

Global dimensions of the Cartesian grid associated to the given domain
decomposition.

Like [`size_local`](@ref), by default the returned dimensions are in logical
order.
"""
size_global(p::Pencil, ::LogicalOrder) = p.size_global
size_global(p::Pencil, ::MemoryOrder) = permutation(p) * p.size_global
size_global(p) = size_global(p, DefaultOrder())

"""
    to_local(p::Pencil, global_inds, [order = LogicalOrder()])

Convert non-permuted (logical) global indices to local indices.

If `order = MemoryOrder()`, returned indices will be permuted using the
permutation associated to the pencil configuration `p`.
"""
function to_local(p::Pencil{N}, global_inds::ArrayRegion{N},
                  order::AbstractIndexOrder = DefaultOrder()) where {N}
    ind = map(global_inds, p.axes_local) do rg, rl
        @assert step(rg) == 1
        δ = 1 - first(rl)
        (first(rg) + δ):(last(rg) + δ)
    end :: ArrayRegion{N}
    order === MemoryOrder() ? (permutation(p) * ind) : ind
end

end
