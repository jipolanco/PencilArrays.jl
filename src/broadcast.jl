using Base.Broadcast: BroadcastStyle, ArrayStyle, Broadcasted

BroadcastStyle(::Type{<:PencilArray}) = ArrayStyle{PencilArray}()
BroadcastStyle(::Type{<:GlobalPencilArray}) = ArrayStyle{PencilArray}()

function Base.similar(bc::Broadcasted{ArrayStyle{PencilArray}},
                      ::Type{T}) where {T}
    A = find_pa(bc)
    similar(A, T, axes(bc))
end

# Find PencilArray or GlobalPencilArray among broadcast arguments.
find_pa(bc::Broadcasted) = find_pa(bc.args...)
find_pa() = nothing
find_pa(A::PencilArray, args...) = A
find_pa(A::GlobalPencilArray, args...) = A
find_pa(::Any, args...) = find_pa(args...)
