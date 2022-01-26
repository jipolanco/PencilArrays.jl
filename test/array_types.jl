# Test PencilArrays wrapping arrays of type different from the base Array.
# We use a custom TestArray type as an example, but this should also
# cover the case of GPU arrays.

using MPI
using PencilArrays
using PencilArrays: typeof_array
using OffsetArrays
using LinearAlgebra: transpose!
using Random
using Test

# Define simple array wrapper type for tests.
struct TestArray{T, N} <: DenseArray{T, N}
    data :: Array{T, N}
end
TestArray{T}(args...) where {T} = TestArray(Array{T}(args...))
TestArray{T,N}(args...) where {T,N} = TestArray(Array{T,N}(args...))
Base.parent(u::TestArray) = u.data
Base.size(u::TestArray) = size(parent(u))
Base.similar(u::TestArray, ::Type{T}, dims::Dims) where {T} =
    TestArray(similar(parent(u), T, dims))
Base.getindex(u::TestArray, args...) = getindex(parent(u), args...)
Base.setindex!(u::TestArray, args...) = (setindex!(parent(u), args...); u)
Base.resize!(u::TestArray, args...) = (resize!(parent(u), args...); u)
Base.pointer(u::TestArray) = pointer(parent(u))
Base.unsafe_wrap(::Type{TestArray}, args...; kws...) =
    TestArray(unsafe_wrap(Array, args...; kws...))
MPI.Buffer(u::TestArray) = MPI.Buffer(parent(u))  # for `gather`

MPI.Init()
comm = MPI.COMM_WORLD
MPI.Comm_rank(comm) == 0 || redirect_stdout(devnull)

@testset "Array type: $A" for A ∈ (Array, TestArray)
    pen = @inferred Pencil(A, (8, 10), comm)
    @test @inferred(typeof_array(pen)) === A
    @test (@inferred (p -> p.send_buf)(pen)) isa A
    @test (@inferred (p -> p.recv_buf)(pen)) isa A

    u = @inferred PencilArray{Int}(undef, pen)
    @test typeof(parent(u)) <: A{Int}

    @test @inferred(PencilArrays.typeof_array(pen)) === A
    @test @inferred(PencilArrays.typeof_array(u)) === A

    @testset "Transpositions" begin
        px = @inferred Pencil(A, (20, 16), (1, ), comm)
        py = @inferred Pencil(px; decomp_dims = (2, ))
        @test @inferred(typeof_array(px)) === A
        @test @inferred(typeof_array(py)) === A
        ux = rand!(PencilArray{Float64}(undef, px))
        uy = @inferred similar(ux, py)
        @test pencil(uy) === py
        tr = @inferred Transpositions.Transposition(uy, ux)
        transpose!(tr)

        # Verify transposition
        gx = @inferred Nothing gather(ux)
        gy = @inferred Nothing gather(uy)
        if !(nothing === gx === gy)
            @test typeof(gx) === typeof(gy) <: Array
            @test gx == gy
        end
    end
end
