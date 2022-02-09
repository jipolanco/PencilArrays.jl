# We force specialisation on each function to avoid (tiny) allocations.
#
# Note that, for mapreduce, we can assume that the operation is commutative,
# which allows MPI to freely reorder operations.
#
# We also define mapfoldl (and mapfoldr) for completeness, even though the global
# operations are not strictly performed from left to right (or from right to
# left), since each process locally reduces first.
for (func, commutative) in [:mapreduce => true, :mapfoldl => false, :mapfoldr => false]
    @eval function Base.$func(
            f::F, op::OP, u::PencilArray, etc::Vararg{PencilArray}; kws...,
        ) where {F, OP}
        foreach(v -> _check_compatible_arrays(u, v), etc)
        comm = get_comm(u)
        ups = map(parent, (u, etc...))
        rlocal = $func(f, op, ups...; kws...)
        op_mpi = MPI.Op(op, typeof(rlocal); iscommutative = $commutative)
        MPI.Allreduce(rlocal, op_mpi, comm)
    end

    # Make things work with zip(u::PencilArray, v::PencilArray, ...)
    @eval function Base.$func(
            f::F, op::OP, z::Iterators.Zip{<:Tuple{Vararg{PencilArray}}}; kws...,
        ) where {F, OP}
        g(args...) = f(args)
        $func(g, op, z.is...; kws...)
    end
end
