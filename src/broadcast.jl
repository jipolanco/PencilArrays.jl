using Base.Broadcast: BroadcastStyle, ArrayStyle, Broadcasted

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

# Make PencilArray and GlobalPencilArray incompatible for broadcasting.
# Without this, broadcasting will work with 1 MPI process, but fail with more
# (with an error by OffsetArrays), which is annoying when testing code.
function BroadcastStyle(
        ::ArrayStyle{GlobalPencilArray},
        ::ArrayStyle{A},
    ) where {A <: AbstractArray}
    throw(ArgumentError("cannot combine $A and GlobalPencilArray in broadcast"))
end

# Find PencilArray or GlobalPencilArray among broadcast arguments.
find_pa(bc::Broadcasted) = find_pa(bc.args...)
find_pa() = nothing
find_pa(::Any, args...) = find_pa(args...)
