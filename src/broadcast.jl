using Base.Broadcast: BroadcastStyle, ArrayStyle, DefaultArrayStyle, Broadcasted

for PA in (PencilArray, GlobalPencilArray)
    @eval begin
        BroadcastStyle(::Type{<:$PA}) = ArrayStyle{$PA}()

        function Base.similar(bc::Broadcasted{ArrayStyle{$PA}}, ::Type{T}) where {T}
            A = find_pa(bc)
            similar(A, T, axes(bc))
        end

        find_pa(A::$PA, args...) = A
    end
end

# For some reason, defining this improves performance a bit, and doesn't change
# the result.
function Base.copyto!(dest::Union{PencilArray,GlobalPencilArray},
                      bc::Broadcasted{Nothing})
    @assert size(dest) === size(bc)
    @inbounds for n in eachindex(dest)
        dest[n] = bc[n]
    end
    dest
end

# Make PencilArray and GlobalPencilArray incompatible for broadcasting.
# Without this, broadcasting will work with 1 MPI process, but fail with more
# (with an error by OffsetArrays), which is annoying when testing code.
function BroadcastStyle(
        ::ArrayStyle{GlobalPencilArray},
        ::ArrayStyle{A},
    ) where {A <: AbstractArray}
    throw(ArgumentError("cannot combine $A and GlobalPencilArray in broadcast"))
end

# For the same reasons, disallow broadcasting between generic array and
# GlobalPencilArray.
function BroadcastStyle(::ArrayStyle{GlobalPencilArray}, ::DefaultArrayStyle)
    throw(ArgumentError("cannot combine generic arrays and GlobalPencilArray in broadcast"))
end

# Exception: broadcasting with scalars.
BroadcastStyle(style::ArrayStyle{GlobalPencilArray}, ::DefaultArrayStyle{0}) = style

# Find PencilArray or GlobalPencilArray among broadcast arguments.
find_pa(bc::Broadcasted) = find_pa(bc.args...)
find_pa(::Any, args...) = find_pa(args...)
