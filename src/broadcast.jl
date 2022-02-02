using Base.Broadcast:
    BroadcastStyle, Broadcasted, AbstractArrayStyle, DefaultArrayStyle

abstract type AbstractPencilArrayStyle{N} <: AbstractArrayStyle{N} end

struct PencilArrayStyle{N} <: AbstractPencilArrayStyle{N} end
struct GlobalPencilArrayStyle{N} <: AbstractPencilArrayStyle{N} end

BroadcastStyle(::Type{<:PencilArray{T, N}}) where {T, N} =
    PencilArrayStyle{N}()

# PencilArrayStyle wins against other array styles
BroadcastStyle(style::AbstractPencilArrayStyle, ::AbstractArrayStyle) = style

# Needed to avoid ambiguities
BroadcastStyle(
    style::AbstractPencilArrayStyle{N}, ::DefaultArrayStyle{N},
) where {N} = style

BroadcastStyle(::Type{<:GlobalPencilArray{T, N}}) where {T, N} =
    GlobalPencilArrayStyle{N}()

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
find_pa(A::Union{PencilArray, GlobalPencilArray}, rest) = A

# Make PencilArray and GlobalPencilArray incompatible for broadcasting.
# Without this, broadcasting will work with 1 MPI process, but fail with more
# (with an error by OffsetArrays), which is annoying when testing code.
function BroadcastStyle(::GlobalPencilArrayStyle, ::PencilArrayStyle)
    throw(ArgumentError(
        "cannot combine PencilArray and GlobalPencilArray in broadcast"
    ))
end

# For the same reasons, disallow broadcasting between generic array and
# GlobalPencilArray.
function BroadcastStyle(::GlobalPencilArrayStyle, ::AbstractArrayStyle)
    throw(ArgumentError("cannot combine generic arrays and GlobalPencilArray in broadcast"))
end

# Exception: broadcasting with scalars.
BroadcastStyle(style::GlobalPencilArrayStyle, ::AbstractArrayStyle{0}) = style
