# Test PencilArrays wrapping arrays of type different from the base Array.
# We use OffsetArrays as an example, but this should also cover the case of GPU arrays.

using MPI
using PencilArrays
using OffsetArrays
using Test

MPI.Init()
comm = MPI.COMM_WORLD
MPI.Comm_rank(comm) == 0 || redirect_stdout(devnull)

make_buffer(::Type{A}) where {A} = A{UInt8}(undef, 0)
make_buffer(; array_type::Type{A}) where {A} = make_buffer(A)

@testset "Array type: $A" for A ∈ (Array, OffsetArray)
    pen = @inferred Pencil(A, (8, 10), comm)
    @test (@inferred (p -> p.send_buf)(pen)) isa A
    @test (@inferred (p -> p.recv_buf)(pen)) isa A

    u = @inferred PencilArray{Int}(undef, pen)
    @test typeof(parent(u)) <: A{Int}

    @test @inferred(PencilArrays.typeof_array(pen)) === A
    @test @inferred(PencilArrays.typeof_array(u)) === A
end
