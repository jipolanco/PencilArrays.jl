# We force specialisation on each function to avoid (tiny) allocations.
function Base.mapreduce(f::F, op::OP, u::PencilArray; kws...) where {F, OP}
    rlocal = mapreduce(f, op, parent(u); kws...)
    MPI.Allreduce(rlocal, op, get_comm(u))
end
