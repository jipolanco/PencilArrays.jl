# Test PencilArrays wrapping arrays of type different from the base Array.
# We use a custom TestArray type as well as the JLArray (<: AbstractGPUArray)
# type defined in the GPUArrays.jl tests.

using MPI
using PencilArrays
using PencilArrays: typeof_array
using Random
using Test

## ================================================================================ ##

using JLArrays: JLArray, DenseJLVector, DataRef

# A bit of type piracy to help tests pass (the following functions seem to be defined for
# CuArray).
function Base.resize!(u::DenseJLVector, n)
    T = eltype(u)
    obj = u.data.rc.obj :: Vector{UInt8}
    u.dims = (n,)
    resize!(obj, n * sizeof(T))
    @assert sizeof(obj) === sizeof(u)
    u
end
function Base.unsafe_wrap(::Type{JLArray}, p::Ptr, dims::Dims; kws...)
    T = eltype(p)
    N = length(dims)
    p_obj = convert(Ptr{UInt8}, p)
    dims_obj = (sizeof(T) * prod(dims),)
    obj = unsafe_wrap(Array, p_obj, dims_obj; kws...)
    ref = DataRef(obj)
    x = JLArray{T,N}(ref, dims)
    @assert pointer(x) === p
    x
end
Base.unsafe_wrap(::Type{JLArray}, p::Ptr, n::Integer; kws...) =
    unsafe_wrap(JLArray, p, (n,); kws...)
# Random.rand!(rng::AbstractRNG, u::JLArray, ::Type{X}) where {X} = (rand!(rng, u.data, X); u)

# For some reason this kind of view doesn't work correctly in the original implementation,
# returning a copy.
function Base.view(u::DenseJLVector, I::AbstractUnitRange)
    a, b = first(I), last(I)
    inds = a:1:b  # this kind of range works correctly
    view(u, inds)
end

# Note that MPI.Buffer is also defined for CuArray.
function MPI.Buffer(u::JLArray)
    obj = u.data.rc.obj :: Vector{UInt8}
    count = length(u)
    datatype = MPI.Datatype(eltype(u))
    MPI.Buffer(obj, count, datatype)
end

## ================================================================================ ##

# Define simple array wrapper type for tests.
struct TestArray{T, N} <: DenseArray{T, N}
    data :: Array{T, N}
end
TestArray{T}(args...) where {T} = TestArray(Array{T}(args...))
TestArray{T,N}(args...) where {T,N} = TestArray(Array{T,N}(args...))
Base.parent(u::TestArray) = u  # this is just for the tests...
Base.size(u::TestArray) = size(u.data)
Base.similar(u::TestArray, ::Type{T}, dims::Dims) where {T} =
    TestArray(similar(u.data, T, dims))
Base.getindex(u::TestArray, args...) = getindex(u.data, args...)
Base.setindex!(u::TestArray, args...) = setindex!(u.data, args...)
Base.resize!(u::TestArray, args...) = (resize!(u.data, args...); u)
Base.pointer(u::TestArray) = pointer(u.data)
Base.pointer(u::TestArray, n::Integer) = pointer(u.data, n)  # needed to avoid ambiguity
Base.unsafe_wrap(::Type{TestArray}, p::Ptr, dims::Union{Integer, Dims}; kws...) =
    TestArray(unsafe_wrap(Array, p, dims; kws...))

MPI.Buffer(u::TestArray) = MPI.Buffer(u.data)  # for `gather`
Base.cconvert(::Type{MPI.MPIPtr}, u::TestArray{T}) where {T} =
    reinterpret(MPI.MPIPtr, pointer(u))

MPI.Init()
comm = MPI.COMM_WORLD
MPI.Comm_rank(comm) == 0 || redirect_stdout(devnull)

@testset "Array type: $A" for A ∈ (JLArray, Array, TestArray)
    pen = @inferred Pencil(A, (8, 10), comm)
    @test @inferred(typeof_array(pen)) === A
    @test (@inferred (p -> p.send_buf)(pen)) isa A
    @test (@inferred (p -> p.recv_buf)(pen)) isa A

    # Check that creating a PencilArray with incorrect type of underlying data
    # fails.
    ArrayOther = A === Array ? TestArray : Array
    let dims = size_local(pen, MemoryOrder())
        data = ArrayOther{Float32}(undef, dims)
        @test_throws ArgumentError PencilArray(pen, data)
    end

    u = @inferred PencilArray{Float32}(undef, pen)
    @test typeof(parent(u)) <: A{Float32}

    @test @inferred(typeof_array(pen)) === A
    @test @inferred(typeof_array(u)) === A

    # This is in particular to test that, for GPU arrays, scalar indexing is not
    # performed and the correct GPU functions are called.
    rng = MersenneTwister(42)
    @testset "Initialisation" begin
        @test_nowarn fill!(u, 4)
        @test_nowarn rand!(rng, u)
        @test_nowarn randn!(rng, u)
    end

    px = @inferred Pencil(A, (20, 16, 4), (1,), comm)

    @testset "Permutation: $perm" for perm ∈ (NoPermutation(), Permutation(2, 3, 1))
        if perm != NoPermutation()
            # Make sure we're testing the more "interesting" case in which the
            # permutation is not its own inverse.
            @assert inv(perm) != perm
        end
        py = @inferred Pencil(px; decomp_dims = (2,), permute = perm)
        @test permutation(py) == perm
        @test @inferred(typeof_array(px)) === A
        @test @inferred(typeof_array(py)) === A

        @testset "Transpositions" begin
            ux = @test_nowarn rand!(rng, PencilArray{Float64}(undef, px))
            uy = @inferred similar(ux, py)
            @test pencil(uy) === py
            tr = @inferred Transpositions.Transposition(uy, ux)
            transpose!(tr)
            @test_logs (:warn, r"is deprecated") MPI.Waitall!(tr)

            # Verify transposition
            gx = @inferred Nothing gather(ux)
            gy = @inferred Nothing gather(uy)
            if !(nothing === gx === gy)
                @test typeof(gx) === typeof(gy) <: Array
                @test gx == gy
            end
        end

        @testset "Multiarrays" begin
            M = @inferred ManyPencilArray{Float32}(undef, px, py)
            randn!(rng, M.data)
            ux = @inferred first(M)
            uy = @inferred last(M)
            uxp = parent(parent(parent(ux)))
            uyp = parent(parent(parent(uy)))
            @test uxp === M.data
            @test uxp === uyp
            @test typeof(uxp) <: A{Float32}
            @test @inferred(Tuple(M)) === (ux, uy)
        end
    end  # permutation
end
