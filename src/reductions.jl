# We force specialisation on each function to avoid (tiny) allocations.
#
# Note that, for mapreduce, we can assume that the operation is commutative,
# which allows MPI to freely reorder operations.
#
# We also define mapfoldl (and mapfoldr) for completeness, even though the global
# operations are not strictly performed from left to right (or from right to
# left), since each process locally reduces first.
for (func, commutative) in [:mapreduce => true, :mapfoldl => false, :mapfoldr => false]
    @eval function Base.$func(f::F, op::OP, u::PencilArray; kws...) where {F, OP}
        rlocal = $func(f, op, parent(u); kws...)
        op_mpi = MPI.Op(op, typeof(rlocal); iscommutative = $commutative)
        MPI.Allreduce(rlocal, op_mpi, get_comm(u))
    end
end
for (func, commutative) in [:mapreduce => true, :mapfoldl => false, :mapfoldr => false]
    @eval function Base.$func(f::F, op::OP, u::PencilArray, v::PencilArray; kws...) where {F, OP}
        rlocal = $func(f, op, parent(u), parent(v); kws...)
        op_mpi = MPI.Op(op, typeof(rlocal); iscommutative = $commutative)
        @assert get_comm(u) == get_comm(v)
        MPI.Allreduce(rlocal, op_mpi, get_comm(u))
    end
end
