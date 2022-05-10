module Pencils

import ..Permutations: permutation
import ..LocalGrids

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
export range_local, range_remote, size_local, size_global, to_local,
       length_local, length_global

# Describes the portion of an array held by a given MPI process.
const ArrayRegion{N} = NTuple{N,UnitRange{Int}} where N

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
        [A = Array],
        topology::MPITopology{M}, size_global::Dims{N},
        decomp_dims::Dims{M} = default_decomposition(N, Val(M));
        permute::AbstractPermutation = NoPermutation(),
        timer = TimerOutput(),
    )

Define the decomposition of an `N`-dimensional geometry along `M` dimensions.

The dimensions of the geometry are given by `size_global = (N1, N2, ...)`. The
`Pencil` describes the decomposition of an array of dimensions `size_global`
across a group of MPI processes.

Data is distributed over the given `M`-dimensional MPI topology (with `M < N`).

The decomposed dimensions may optionally be provided via the `decomp_dims`
argument. By default, the `M` rightmost dimensions are decomposed. For instance,
for a 2D decomposition of 5D data (`M = 2` and `N = 5`), the dimensions `(4, 5)`
are decomposed by default.

The optional argument `A` allows to work with arrays other than the base `Array`
type. In particular, this should be useful for working with GPU array types such
as `CuArray`.

The optional `permute` parameter may be used to indicate a permutation of the
data indices from **logical order** (the order in which the
arrays are accessed in code) to **memory order** (the actual order of indices in
memory). Permutations must be specified using the exported `Permutation` type,
as in `permute = Permutation(3, 1, 2)`.

It is also possible to pass a `TimerOutput` to the constructor. See
[Measuring performance](@ref PencilArrays.measuring_performance) for details.

# Examples

Decompose a 3D geometry of global dimensions ``N_x × N_y × N_z = 4×8×12`` along
the second (``y``) and third (``z``) dimensions:

```jldoctest
julia> topo = MPITopology(MPI.COMM_WORLD, Val(2));

julia> Pencil(topo, (4, 8, 12), (2, 3))
Decomposition of 3D data
    Data dimensions: (4, 8, 12)
    Decomposed dimensions: (2, 3)
    Data permutation: NoPermutation()
    Array type: Array

julia> Pencil(topo, (4, 8, 12), (2, 3); permute = Permutation(3, 2, 1))
Decomposition of 3D data
    Data dimensions: (4, 8, 12)
    Decomposed dimensions: (2, 3)
    Data permutation: Permutation(3, 2, 1)
    Array type: Array

```

In the second case, the actual data is stored in `(z, y, x)` order within
each MPI process.

---

    Pencil([A = Array], size_global::Dims{N}, [decomp_dims = (2, …, N)], comm::MPI.Comm; kws...)

Convenience constructor that implicitly creates a [`MPITopology`](@ref).

The number of decomposed dimensions specified by `decomp_dims` must be `M < N`.
If `decomp_dims` is not passed, dimensions `2:N` are decomposed.

Keyword arguments are passed to alternative constructor taking an `MPITopology`.
That constructor should be used if more control is desired.

# Examples

```jldoctest
julia> Pencil((4, 8, 12), MPI.COMM_WORLD)
Decomposition of 3D data
    Data dimensions: (4, 8, 12)
    Decomposed dimensions: (2, 3)
    Data permutation: NoPermutation()
    Array type: Array

julia> Pencil((4, 8, 12), (1, ), MPI.COMM_WORLD)
Decomposition of 3D data
    Data dimensions: (4, 8, 12)
    Decomposed dimensions: (1,)
    Data permutation: NoPermutation()
    Array type: Array
```

---

    Pencil(
        [A = Array],
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
        BufVector <: AbstractVector{UInt8},
    }
    # M-dimensional MPI decomposition info (with M < N).
    topology :: MPITopology{M}

    # Global array dimensions (N1, N2, ...) in logical order.
    # These dimensions are *before* permutation by perm.
    size_global :: Dims{N}

    # Decomposition directions.
    # Example: for x-pencils, this is typically (2, 3, ..., N).
    # Note that don't need to be sorted.
    # The order matters when determining over how many processes a given
    # dimension is distributed.
    # This is in particular important for determining whether two Pencil's are
    # compatible for transposing between them.
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
    send_buf :: BufVector
    recv_buf :: BufVector

    # Timing information.
    timer :: TimerOutput

    function check_empty_dimension(topology, size_global, decomp_dims)
        proc_dims = size(topology)
        for (i, nproc) ∈ zip(decomp_dims, proc_dims)
            # Check that dimension `i` (which has size `N = size_global[i]`) is
            # being decomposed over a number of processes ≤ N.
            N = size_global[i]
            nproc ≤ N && continue
            @warn(
                """
                Dimension `i = $i` has global size `Nᵢ = $N` but is being decomposed across `Pᵢ = $nproc`
                processes.

                Since `Pᵢ > Nᵢ`, some processes will have no data, and therefore will do no work. This can
                result in broadcasting errors and other unsupported behaviour!

                To fix this, consider choosing a different configuration of processes (e.g. via the
                `proc_dims` argument), or use a lower number of processes. See below for the current
                values of some of these parameters.

                """,
                i, size_global, decomp_dims, proc_dims,
            )
            return
        end
        nothing
    end

    # This constructor is left undocumented and should never be called directly.
    global function _Pencil(
            topology::MPITopology{M}, size_global::Dims{N},
            decomp_dims::Dims{M}, axes_all, perm::P,
            send_buf::BufVector, recv_buf::BufVector, timer::TimerOutput,
        ) where {M, N, P, BufVector}
        check_permutation(perm)
        check_empty_dimension(topology, size_global, decomp_dims)
        axes_local = axes_all[coords_local(topology)...]
        axes_local_perm = perm * axes_local
        new{N, M, P, BufVector}(
            topology, size_global, decomp_dims,
            axes_all, axes_local, axes_local_perm,
            perm, send_buf, recv_buf, timer,
        )
    end

    function Pencil(
            topology::MPITopology{M}, size_global::Dims{N},
            decomp_dims::Dims{M} = default_decomposition(N, Val(M));
            permute::AbstractPermutation = NoPermutation(),
            send_buf = UInt8[], recv_buf = UInt8[],
            timer = TimerOutput(),
        ) where {N, M}
        _check_selected_dimensions(N, decomp_dims)
        axes_all = get_axes_matrix(decomp_dims, topology.dims, size_global)
        _Pencil(
            topology, size_global, decomp_dims, axes_all, permute,
            send_buf, recv_buf, timer,
        )
    end

    function Pencil(
            p::Pencil{N,M};
            decomp_dims::Dims{M} = decomposition(p),
            size_global::Dims{N} = size_global(p),
            permute = permutation(p),
            timer::TimerOutput = timer(p),
            etc...,
    ) where {N, M}
        Pencil(
            p.topology, size_global, decomp_dims;
            permute=permute, timer=timer,
            send_buf=p.send_buf, recv_buf=p.recv_buf,
            etc...,
        )
    end
end

function Pencil(dims::Dims, decomp::Dims{M}, comm::MPI.Comm; kws...) where {M}
    topo = MPITopology(comm, Val(M))
    Pencil(topo, dims, decomp; kws...)
end

Pencil(dims::Dims{N}, comm::MPI.Comm; kws...) where {N} =
    Pencil(dims, default_decomposition(N, Val(N - 1)), comm; kws...)

function Pencil(::Type{A}, args...; kws...) where {A <: AbstractArray}
    # We initialise the array with a single element to work around problem
    # with CuArrays: if its length is zero, then the CuArray doesn't have a
    # valid pointer.
    send_buf = A{UInt8}(undef, 1)
    Pencil(args...; kws..., send_buf, recv_buf = similar(send_buf))
end

# Strips array type:
#   Array{Int, 3} -> Array
#   Array{Int}    -> Array
#   Array         -> Array
@generated function typeof_array(::Type{A′}) where {A′ <: AbstractArray}
    A = A′
    while A isa UnionAll
        A = A.body
    end
    T = A.name.wrapper
    :($T)
end

typeof_array(A::AbstractArray) = typeof_array(typeof(A))
typeof_array(p::Pencil) = typeof_array(p.send_buf)

"""
    similar(p::Pencil, [A = typeof_array(p)], [dims = size_global(p)])

Returns a [`Pencil`](@ref) decomposition with global dimensions `dims` and with
underlying array type `A`.

Typically, `A` should be something like `Array` or `CuArray` (see
[`Pencil`](@ref) for details).
"""
function Base.similar(
        p::Pencil{N}, ::Type{A}, dims::Dims{N} = size_global(p),
    ) where {A <: AbstractArray, N}
    _similar(A, typeof_array(p), p, dims)
end

Base.similar(p::Pencil{N}, dims::Dims{N} = size_global(p)) where {N} =
    similar(p, typeof_array(p), dims)

# Case A === A′
function _similar(
        ::Type{A}, ::Type{A}, p::Pencil{N}, dims::Dims{N},
    ) where {N, A <: AbstractArray}
    @assert typeof_array(p) === A
    if dims == size_global(p)
        p  # avoid all copies
    else
        Pencil(p; size_global = dims)
    end
end

# Case A !== A′ (→ change of array type)
function _similar(
        ::Type{A′}, ::Type{A}, p::Pencil{N}, dims::Dims{N},
    ) where {N, A <: AbstractArray, A′ <: AbstractArray}
    @assert typeof_array(p) === A

    # We initialise the array with a single element to work around problem
    # with CuArrays: if its length is zero, then the CuArray doesn't have a
    # valid pointer.
    send_buf = A′{UInt8}(undef, 1)
    recv_buf = similar(send_buf)

    if dims == size_global(p)
        # Avoid recomputing (and allocating a new) `axes_all`, since it doesn't
        # change in the new decomposition.
        _Pencil(
            p.topology, dims, p.decomp_dims, p.axes_all,
            p.perm, send_buf, recv_buf, p.timer,
        )
    else
        Pencil(
            p.topology, dims, p.decomp_dims;
            permute = p.perm, send_buf, recv_buf, timer = p.timer,
        )
    end
end

function check_permutation(perm)
    isperm(perm) && return
    throw(ArgumentError("invalid permutation of dimensions: $perm"))
end

function default_decomposition(N, ::Val{M}) where {M}
    @assert 0 < M ≤ N
    ntuple(d -> N - M + d, Val(M))
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

Base.summary(io::IO, p::Pencil) = Base.showarg(io, p, true)

function Base.showarg(io::IO, p::Pencil{N,M,P}, toplevel) where {N,M,P}
    toplevel || print(io, "::")
    A = typeof_array(p)
    print(io, nameof(typeof(p)), "{$N, $M, $P, $A}")
end

function Base.show(io::IO, p::Pencil)
    perm = permutation(p)
    print(io,
          """
          Decomposition of $(ndims(p))D data
              Data dimensions: $(size_global(p))
              Decomposed dimensions: $(decomposition(p))
              Data permutation: $(perm)
              Array type: $(typeof_array(p))""")
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

Get linear length of the *local* data associated to the decomposition.

Equivalent to `length_local(p)`.
"""
Base.length(p::Pencil) = prod(size(p))

"""
    length_local(p::Pencil)

Get linear length of the local data associated to the decomposition.
"""
length_local(p::Pencil) = prod(size_local(p))

"""
    length_global(p::Pencil)

Get linear length of the global data associated to the decomposition.
"""
length_global(p::Pencil) = prod(size_global(p))

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
    size(p::Pencil)

Returns the *local* data dimensions associated to the decomposition, in *logical*
order.

This is defined as `size_local(p, LogicalOrder())`.
"""
Base.size(p::Pencil) = size_local(p, LogicalOrder())

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

"""
    localgrid(p::Pencil, (x_global, y_global, ...))      -> LocalRectilinearGrid
    localgrid(u::PencilArray, (x_global, y_global, ...)) -> LocalRectilinearGrid

Create a [`LocalRectilinearGrid`](@ref LocalGrids.LocalRectilinearGrid) from a
decomposition configuration and from a set of orthogonal global coordinates
`(x_global, y_global, ...)`.

In this case, each `*_global` is an `AbstractVector` describing the coordinates
along one dimension of the global grid.
"""
function LocalGrids.localgrid(p::Pencil, coords_global::Tuple{Vararg{AbstractVector}})
    perm = permutation(p)
    ranges = range_local(p, LogicalOrder())
    coords_local = map(view, coords_global, ranges)
    LocalGrids.localgrid(coords_local, perm)
end

end
