# The broadcasting logic is quite tricky due to possible dimension permutations.
# The basic idea is that, when permutations are enabled, PencilArrays broadcast
# using their dimensions in memory order.
# This allows things like `u .+ parent(u)` even when `u` is a PencilArray with
# permuted dimensions.
# In this case, `u` and `parent(u)` may have different sizes [e.g. `(4, 6, 7)`
# vs `(6, 7, 4)`] but they're still allowed to broadcast, which may not be very
# natural or intuitive.

using Base.Broadcast:
    Broadcast,
    BroadcastStyle, Broadcasted,
    AbstractArrayStyle, DefaultArrayStyle

abstract type AbstractPencilArrayStyle{N} <: AbstractArrayStyle{N} end

struct PencilArrayStyle{N} <: AbstractPencilArrayStyle{N} end
struct GlobalPencilArrayStyle{N} <: AbstractPencilArrayStyle{N} end

struct PencilArrayBroadcastable{T, N, A <: Union{PencilArray, GlobalPencilArray}}
    data :: A
    PencilArrayBroadcastable(u::AbstractArray{T, N}) where {T, N} =
        new{T, N, typeof(u)}(u)
end

_actual_parent(u::PencilArray) = parent(u)
_actual_parent(u::GlobalPencilArray) = parent(parent(u))
_actual_parent(bc::PencilArrayBroadcastable) = _actual_parent(bc.data)

Broadcast.broadcastable(x::Union{PencilArray, GlobalPencilArray}) =
    PencilArrayBroadcastable(x)

Base.axes(bc::PencilArrayBroadcastable{Union{PencilArray, GlobalPencilArray}}) =
    axes(_actual_parent(bc))
Base.ndims(::Type{PencilArrayBroadcastable{T, N}}) where {T, N} = N
Base.size(bc::PencilArrayBroadcastable) = size(_actual_parent(bc))
Base.@propagate_inbounds Base.getindex(bc::PencilArrayBroadcastable, inds...) =
    _actual_parent(bc)[inds...]
Base.@propagate_inbounds Base.setindex!(bc::PencilArrayBroadcastable, args...) =
    setindex!(_actual_parent(bc), args...)
Base.similar(bc::PencilArrayBroadcastable, ::Type{T}) where {T} =
    PencilArrayBroadcastable(similar(bc.data, T))

function Broadcast.materialize!(
        u::Union{PencilArray, GlobalPencilArray},
        bc::Broadcasted,
    )
    Broadcast.materialize!(_actual_parent(u), bc)
    u
end

function Broadcast.materialize(bc::Broadcasted{<:AbstractPencilArrayStyle})
    u = copy(Broadcast.instantiate(bc)) :: PencilArrayBroadcastable
    u.data
end

BroadcastStyle(::Type{<:PencilArrayBroadcastable{T, N, <:PencilArray}}) where {T, N} =
    PencilArrayStyle{N}()
BroadcastStyle(::Type{<:PencilArrayBroadcastable{T, N, <:GlobalPencilArray}}) where {T, N} =
    GlobalPencilArrayStyle{N}()

# AbstractPencilArrayStyle wins against other array styles
BroadcastStyle(style::AbstractPencilArrayStyle, ::AbstractArrayStyle) = style

# This is needed to avoid ambiguities
BroadcastStyle(style::AbstractPencilArrayStyle, ::DefaultArrayStyle) = style

# TODO can this be allowed?
# Make PencilArray and GlobalPencilArray incompatible for broadcasting.
# Without this, broadcasting will work with 1 MPI process, but fail with more
# (with an error by OffsetArrays), which is annoying when testing code.
function BroadcastStyle(::GlobalPencilArrayStyle, ::PencilArrayStyle)
    throw(ArgumentError(
        "cannot combine PencilArray and GlobalPencilArray in broadcast"
    ))
end

function Base.similar(
        bc::Broadcasted{<:AbstractPencilArrayStyle}, ::Type{T},
    ) where {T}
    A = find_pa(bc)
    if axes(bc) != axes(A)
        throw(DimensionMismatch("arrays cannot be broadcast; got axes $(axes(bc)) and $(axes(A))"))
    end
    similar(A, T)
end

# Find PencilArray or GlobalPencilArray among broadcast arguments.
find_pa(bc::Broadcasted) = find_pa(bc.args)
find_pa(args::Tuple) = find_pa(find_pa(args[1]), Base.tail(args))
find_pa(x) = x
find_pa(::Any, rest) = find_pa(rest)
find_pa(A::PencilArrayBroadcastable, rest) = A

# When materialising the broadcast, we unwrap all arrays wrapped by PencilArrays.
# This is to make sure that the right `copyto!` is called.
# For GPU arrays, this enables the use of the `copyto!` implementation in
# GPUArrays.jl, avoiding scalar indexing.
function Base.copyto!(dest_in::PencilArray, bc_in::Broadcasted{Nothing})
    dest = _unwrap_pa(dest_in)
    perm = permutation(dest_in)
    bc = _unwrap_pa(bc_in, perm)
    @show bc
    copyto!(dest, bc)
    dest_in
end

function _unwrap_pa(bc::Broadcasted{Nothing}, perm)
    args = map(_unwrap_pa, bc.args)
    axs = perm * axes(bc)  # apply possible permutation
    Broadcasted{Nothing}(bc.f, args, axs)
end

_unwrap_pa(u::PencilArray) = parent(u)
_unwrap_pa(u::GlobalPencilArray) = _unwrap_pa(parent(u))
_unwrap_pa(u) = u

function Base.copyto!(dest::GlobalPencilArray, bc::Broadcasted{Nothing})
    copyto!(parent(dest), bc)
    dest
end
