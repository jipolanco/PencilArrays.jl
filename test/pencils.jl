using PencilArrays
using PencilArrays.MPITopologies

using MPI

using BenchmarkTools
using LinearAlgebra: transpose!
using Random
using Test

const BENCHMARK_ARRAYS = "--benchmark" in ARGS

Indexation(::Type{IndexLinear}) = LinearIndices
Indexation(::Type{IndexCartesian}) = CartesianIndices

# For testing `similar` for PencilArray.
struct DummyArray{T,N} <: AbstractArray{T,N}
    dims :: Dims{N}
    DummyArray{T}(::UndefInitializer, dims::Dims) where {T} = new{T,length(dims)}(dims)
end
DummyArray{T}(init, dims...) where {T} = DummyArray{T}(init, dims)
Base.size(x::DummyArray) = x.dims
Base.getindex(::DummyArray{T}, ind...) where {T} = zero(T)
Base.similar(x::DummyArray, ::Type{S}, dims::Dims) where {S} = DummyArray{S}(undef, dims)

function benchmark_fill!(::Type{T}, u, val) where {T <: IndexStyle}
    indices = Indexation(T)(u)
    @inbounds for I in indices
        u[I] = val
    end
    u
end

function test_array_wrappers(p::Pencil, ::Type{T} = Float64) where {T}
    u = PencilArray{T}(undef, p)

    @test match(r"PencilArray{.*}\(::Pencil{.*}\)", summary(u)) !== nothing

    for x in (42, 10)
        fill!(u, x)
        @test all(==(x), u)
    end

    perm = permutation(u)
    @test perm === permutation(typeof(u))
    let topo = topology(p)
        @test topo === topology(u)
        I = coords_local(topo)
        @test range_remote(u, I) === range_remote(p, I)
    end

    @test parent(u) === u.data
    @test eltype(u) === eltype(u.data) === T
    @test length.(axes(u)) === size_local(u)
    @test sizeof_global(u) == sizeof(T) * prod(size_global(u))
    @test length_global(u) == prod(size_global(u))
    @test sizeof_global((u, u)) == 2 * sizeof_global(u)
    let umat = [u for i = 1:2, j = 1:3]
        @test sizeof_global(umat) == 6 * sizeof_global(u)
    end

    @test length_global(u) == length_global(p)
    @test size_global(u) == size_global(p)
    @test length_local(u) == length_local(p)
    @test size_local(u) == size_local(p)

    let
        A = PermutedDimsArray(parent(u), perm)
        @test strides(A) === strides(u)
    end

    randn!(u)
    @test check_iteration_order(u)

    @inferred global_view(u)
    ug = global_view(u)
    @test check_iteration_order(ug)

    @test length(u) == length_local(u)
    @test size(u) == size_local(u)

    if BENCHMARK_ARRAYS
        for S in (IndexLinear, IndexCartesian)
            @info("Filling arrays using $S (Array, PencilArray, GlobalPencilArray)",
                  permutation(p))
            for v in (parent(u), u, ug)
                val = 3 * oneunit(eltype(v))
                @btime benchmark_fill!($S, $v, $val)
            end
            println()
        end
    end

    @testset "similar" begin
        let v = @inferred similar(u)
            @test typeof(v) === typeof(u)
            @test length(v) == length(u)
            @test length_local(v) == length_local(u)
            @test size(v) == size(u)
            @test size_local(v) == size_local(u)
            @test pencil(v) === pencil(u)
        end

        let v = @inferred similar(u, Int)
            @test v isa PencilArray
            @test size_local(v) == size_local(u)
            @test eltype(v) === Int
            @test pencil(v) === pencil(u)
        end

        let v = @inferred similar(u, (3, 4))
            @test v isa Matrix
            @test size(v) == (3, 4)
            @test eltype(v) === eltype(u)
        end

        let v = @inferred similar(u, Int, (3, 4))
            @test v isa Matrix
            @test size(v) == (3, 4)
            @test eltype(v) === Int
        end

        let A = DummyArray{Int}(undef, size_local(p, MemoryOrder()))
            pdummy = Pencil(DummyArray, p)
            local u = @inferred PencilArray(pdummy, A)
            @test parent(u) === A

            v = @inferred similar(u)
            @test typeof(v) === typeof(u)
            @test size(v) == size(u)

            w = @inferred similar(u, (4, 2))
            @test w isa DummyArray{Int,2}
            @test size(w) == (4, 2)

            z = @inferred similar(u, Float32, (3, 4, 2))
            @test z isa DummyArray{Float32,3}
            @test size(z) == (3, 4, 2)
        end

        # Test similar(u, [T], q::Pencil)
        let N = ndims(p)
            permute = Permutation(N, ntuple(identity, N - 1)...)  # = (N, 1, 2, ..., N - 1)
            decomp_dims = mod1.(decomposition(p) .+ 1, N)
            q = Pencil(p; decomp_dims = decomp_dims, permute = permute)

            v = @inferred similar(u, q)
            @test pencil(v) === q
            @test eltype(v) === eltype(u)
            @test size_global(v) === size_global(u)

            w = @inferred similar(u, Int, q)
            @test pencil(w) === q
            @test eltype(w) === Int
            @test size_global(w) === size_global(u)
        end
    end

    @test fill!(u, 42) === u

    let z = @inferred zero(u)
        @test all(iszero, z)
        @test typeof(z) === typeof(u)
        @test pencil(z) === pencil(u)
        @test size(z) === size(u)
        @test size_local(z) === size_local(u)
    end

    let v = similar(u)
        @test typeof(v) === typeof(u)

        psize = size_local(p, LogicalOrder())
        @test psize === size_local(v) === size_local(u)
        @test psize === size_local(u, LogicalOrder()) === size_local(v, LogicalOrder())

        vp = parent(v)
        randn!(vp)
        I = size_local(v) .>> 1  # non-permuted indices
        J = perm * I
        @test v[I...] == vp[J...]  # the parent takes permuted indices
    end

    let psize = size_local(p, MemoryOrder())
        a = zeros(T, psize)
        u = PencilArray(p, a)
        @test parent(u) === a
        @test IndexStyle(typeof(u)) === IndexStyle(typeof(a)) === IndexLinear()

        b = zeros(T, psize .+ 2)
        @test_throws DimensionMismatch PencilArray(p, b)
        @test_throws DimensionMismatch PencilArray(p, zeros(T, 3, psize...))

        # This is allowed.
        w = PencilArray(p, zeros(T, psize..., 3))
        @test size_global(w) === (size_global(p)..., 3)

        @inferred PencilArray(p, zeros(T, psize..., 3))
        @inferred size_global(w)
    end

    nothing
end

function test_multiarrays(pencils::Vararg{Pencil,M};
                          element_type::Type{T} = Float64) where {M,T}
    @assert M >= 3
    @inferred ManyPencilArray{T}(undef, pencils...)

    A = ManyPencilArray{T}(undef, pencils...)

    @test ndims(A) === ndims(first(pencils))
    @test eltype(A) === T
    @test length(A) === M

    @inferred first(A)
    @inferred last(A)
    @inferred A[Val(2)]
    @inferred A[Val(M)]

    @test_throws ErrorException @inferred A[2]  # type not inferred

    @test A[Val(1)] === first(A) === A[Val(UInt8(1))] === A[1]
    @test A[Val(2)] === A[2] === A.arrays[2] === A[Val(Int32(2))]
    @test A[Val(M)] === last(A)

    @test_throws BoundsError A[Val(0)]
    @test_throws BoundsError A[Val(M + 1)]

    @testset "In-place extra dimensions" begin
        e = (3, 2)
        @inferred ManyPencilArray{T}(undef, pencils...; extra_dims=e)
        A = ManyPencilArray{T}(undef, pencils...; extra_dims=e)
        @test extra_dims(first(A)) === extra_dims(last(A)) === e
        @test ndims_extra(first(A)) == ndims_extra(last(A)) == length(e)
    end

    @testset "In-place transpose" begin
        u = A[Val(1)]
        v = A[Val(2)]
        w = A[Val(3)]

        randn!(u)
        u_orig = copy(u)

        transpose!(v, u)  # this also modifies `u`!
        @test compare_distributed_arrays(u_orig, v)

        # In the 1D decomposition case, this is a local transpose, since v and w
        # only differ in the permutation.
        transpose!(w, v)
        @test compare_distributed_arrays(u_orig, w)
    end

    nothing
end

function check_iteration_order(u::Union{PencilArray,GlobalPencilArray})
    p = parent(parent(u)) :: Array  # two `parent` are needed for GlobalPencilArray
    cart = CartesianIndices(u)
    lin = LinearIndices(u)

    # Check that the behaviour of `cart` is consistent with that of
    # CartesianIndices.
    @assert size(CartesianIndices(p)) == size(p)
    @test size(cart) == size_local(u)

    # Same for `lin`.
    @assert size(LinearIndices(p)) == size(p)
    @test size(lin) == size_local(u)

    # Check that Cartesian indices iterate in memory order.
    for (n, I) in enumerate(cart)
        l = lin[I]
        @assert l == n
        u[n] == p[n] == u[I] == u[l] || return false
    end

    # Also test iteration on LinearIndices and their conversion to Cartesian
    # indices.
    for (n, l) in enumerate(lin)
        @assert l == n
        # Convert linear to Cartesian index.
        I = cart[l]  # this is relatively slow, don't do it in real code!
        u[n] == p[n] == u[I] == u[l] || return false
    end

    N = ndims(u)
    @test ndims(lin) == ndims(cart) == N

    true
end

function compare_distributed_arrays(u_local::PencilArray, v_local::PencilArray)
    comm = get_comm(u_local)
    root = 0
    myrank = MPI.Comm_rank(comm)

    u = gather(u_local, root)
    v = gather(v_local, root)

    same = Ref(false)
    if u !== nothing && v !== nothing
        @assert myrank == root
        same[] = u == v
    end
    MPI.Bcast!(same, root, comm)

    same[]
end

MPI.Init()

Nxyz = (16, 21, 41)
comm = MPI.COMM_WORLD
Nproc = MPI.Comm_size(comm)
myrank = MPI.Comm_rank(comm)

MPI.Comm_rank(comm) == 0 || redirect_stdout(devnull)

rng = MersenneTwister(42 + myrank)

# Let MPI_Dims_create choose the values of (P1, P2).
proc_dims = MPITopologies.dims_create(comm, Val(2))

# Note that using dims_create is the default in MPITopology
@test MPITopology(comm, proc_dims) == MPITopology(comm, Val(2))

@test_throws ArgumentError MPITopology(comm, proc_dims .- 1)
@test_throws ArgumentError MPITopology(comm, proc_dims .+ 1)
topo = MPITopology(comm, proc_dims)
@test match(
    r"^MPI topology: 2D decomposition \(\d+×\d+ processes\)$",
    string(topo),
) !== nothing
@test ndims(topo) == length(proc_dims) == 2

pen1 = @inferred Pencil(topo, Nxyz)
let p = @inferred Pencil(topo, Nxyz, (2, 3))  # this is the default decomposition
    @test decomposition(p) === decomposition(pen1)
end
pen2 = @inferred Pencil(pen1, decomp_dims=(1, 3), permute=Permutation(2, 3, 1))
pen3 = @inferred Pencil(pen2, decomp_dims=(1, 2), permute=Permutation(3, 2, 1))

@test match(r"Pencil{3, 2, NoPermutation, Array}", summary(pen1)) !== nothing
@test match(r"Pencil{3, 2, Permutation{.*}, Array}", summary(pen2)) !== nothing

println("Pencil 1: ", pen1, "\n")
println("Pencil 2: ", pen2, "\n")
println("Pencil 3: ", pen3, "\n")

@testset "Pencil constructors" begin
    comm = MPI.COMM_WORLD

    p = @inferred Pencil((5, 4, 4, 3), comm)
    @test decomposition(p) == (2, 3, 4)
    @test ndims(topology(p)) == 3
    @test permutation(p) == NoPermutation()

    p = @inferred Pencil((5, 4, 4, 3), comm;
                         permute = Permutation(2, 3, 4, 1))
    @test decomposition(p) == (2, 3, 4)
    @test ndims(topology(p)) == 3
    @test permutation(p) == Permutation(2, 3, 4, 1)

    p = @inferred Pencil((5, 4, 4, 3), (2, 3), comm;
                         permute = Permutation(2, 3, 4, 1))
    @test decomposition(p) == (2, 3)
    @test ndims(topology(p)) == 2
    @test permutation(p) == Permutation(2, 3, 4, 1)
end

@testset "ManyPencilArray" begin
    test_multiarrays(pen1, pen2, pen3)
end

@testset "Topology" begin
    @test topology(pen2) === topo
    @test range_remote(pen2, coords_local(topo)) == range_local(pen2)
    @test eachindex(topo) isa LinearIndices
    for (n, I) in zip(eachindex(topo), CartesianIndices(topo))
        for order in (MemoryOrder(), LogicalOrder())
            @test range_remote(pen2, Tuple(I), order) ==
                range_remote(pen2, n, order)
        end
    end
end

# Note: the permutation of pen2 was chosen such that the inverse permutation
# is different.
@assert permutation(pen2) != inv(permutation(pen2))

@testset "Pencil constructor checks" begin
    # Too many decomposed directions
    @test_throws ArgumentError Pencil(
        MPITopology(comm, (Nproc, 1, 1)), Nxyz, (1, 2, 3))

    # Invalid permutations
    @test_throws TypeError Pencil(
        topo, Nxyz, (1, 2), permute=(2, 3, 1))
    @test_throws ArgumentError Pencil(
        topo, Nxyz, (1, 2), permute=Permutation(0, 3, 15))

    # Decomposed dimensions may not be repeated.
    @test_throws ArgumentError Pencil(topo, Nxyz, (2, 2))

    # Decomposed dimensions must be in 1:N = 1:3.
    @test_throws ArgumentError Pencil(topo, Nxyz, (1, 4))
    @test_throws ArgumentError Pencil(topo, Nxyz, (0, 2))

    @test Pencils.complete_dims(Val(5), (2, 3), (42, 12)) ===
        (1, 42, 12, 1, 1)

    # Divide dimension of size = Nproc - 1 among Nproc processes.
    # => One process will have no data!
    global_dims = (12, Nproc - 1)
    decomp_dims = (2,)
    proc = MPI.Comm_rank(comm) + 1
    js_local = Pencils.local_data_range(proc, Nproc, global_dims[2])
    @test MPI.Allreduce(length(js_local), +, comm) == global_dims[2]
    if isempty(js_local)  # if this process has no data
        @test_warn "has no data" Pencil(global_dims, decomp_dims, comm)
    end
end

@testset "Pencil" begin
    for p ∈ (pen1, pen2, pen3)
        @test size(p) === size_local(p, LogicalOrder())
        @test length(p) === prod(size(p))
        @inferred (p -> p.send_buf)(p)
    end

    @testset "similar" begin
        p = pen2
        # Case 1a: identical pencil
        let q = @inferred similar(p)
            @test q === p
        end
        # Case 1b: different dimensions
        let q = @inferred similar(p, 2 .* size_global(p))
            @test size_global(q) == 2 .* size_global(p)
        end
        # Case 2a: same dimensions but different array type
        let q = @inferred similar(p, DummyArray)
            @test q !== p
            @test q.axes_all === p.axes_all  # array wasn't copied nor recomputed
            @test size_global(q) == size_global(p)
            @test Pencils.typeof_array(q) === DummyArray
        end
        # Case 2b: different dimensions and array type
        let q = @inferred similar(p, DummyArray, 2 .* size_global(p))
            @test q !== p
            @test size_global(q) == 2 .* size_global(p)
            @test Pencils.typeof_array(q) === DummyArray
        end
    end
end

@testset "PencilArray" begin
    test_array_wrappers(pen1)
    test_array_wrappers(pen2)
    test_array_wrappers(pen3)
end

transpose_methods = (Transpositions.PointToPoint(),
                     Transpositions.Alltoallv())

@testset "transpose! $method" for method in transpose_methods
    T = Float64
    u1 = PencilArray{T}(undef, pen1)
    u2 = PencilArray{T}(undef, pen2)
    u3 = PencilArray{T}(undef, pen3)

    # Set initial random data.
    randn!(rng, u1)
    u1 .+= 10 * myrank
    u1_orig = copy(u1)

    # Direct u1 -> u3 transposition is not possible!
    @test_throws ArgumentError transpose!(u3, u1, method=method)

    # Transpose back and forth between different pencil configurations
    transpose!(u2, u1, method=method)
    @test compare_distributed_arrays(u1, u2)

    transpose!(u3, u2, method=method)
    @test compare_distributed_arrays(u2, u3)

    transpose!(u2, u3, method=method)
    @test compare_distributed_arrays(u2, u3)

    transpose!(u1, u2, method=method)
    @test compare_distributed_arrays(u1, u2)

    @test u1_orig == u1

    # Test transpositions without permutations.
    let pen2 = Pencil(pen1, decomp_dims=(1, 3))
        u2 = PencilArray{T}(undef, pen2)
        transpose!(u2, u1, method=method)
        @test compare_distributed_arrays(u1, u2)
    end

end

# Test arrays with extra dimensions.
@testset "extra dimensions" begin
    T = Float32
    u1 = PencilArray{T}(undef, pen1, 3, 4)
    u2 = PencilArray{T}(undef, pen2, 3, 4)
    u3 = PencilArray{T}(undef, pen3, 3, 4)
    @test range_local(u2) ===
        (range_local(pen2)..., Base.OneTo.((3, 4))...)
    @test range_remote(u2, 1) ===
        (range_remote(pen2, 1)..., Base.OneTo.((3, 4))...)
    randn!(rng, u1)
    transpose!(u2, u1)
    @test compare_distributed_arrays(u1, u2)
    transpose!(u3, u2)
    @test compare_distributed_arrays(u2, u3)

    for v in (u1, u2, u3)
        @test check_iteration_order(v)
    end

    @inferred global_view(u1)
end

# Test slab (1D) decomposition.
@testset "1D decomposition" begin
    T = Float32
    topo = MPITopology(comm, (Nproc, ))
    @test ndims(topo) == 1

    pen1 = Pencil(topo, Nxyz, (1, ))
    pen2 = Pencil(pen1, decomp_dims=(2, ))

    # Same decomposed dimension as pen2, but different permutation.
    pen3 = Pencil(pen2, permute=Permutation(3, 2, 1))

    u1 = PencilArray{T}(undef, pen1)
    u2 = @inferred similar(u1, pen2)
    u3 = @inferred similar(u1, pen3)

    @test pencil(u2) === pen2
    @test pencil(u3) === pen3

    randn!(rng, u1)
    transpose!(u2, u1)
    @test compare_distributed_arrays(u1, u2)

    transpose!(u3, u2)
    @test compare_distributed_arrays(u1, u3)
    @test check_iteration_order(u3)

    # Test transposition between two identical configurations.
    transpose!(u2, u2)
    @test compare_distributed_arrays(u1, u2)

    let v = similar(u2)
        @test pencil(u2) === pencil(v)
        transpose!(v, u2)
        @test compare_distributed_arrays(u1, v)
    end

    test_multiarrays(pen1, pen2, pen3)
end

@testset "Inference" begin
    periods = zeros(Int, length(proc_dims))
    comm_cart = MPI.Cart_create(comm, collect(proc_dims), periods, false)
    @inferred MPITopologies.create_subcomms(Val(2), comm_cart)
    @test_throws ArgumentError MPITopology{3}(comm_cart)  # wrong dimensionality
    @inferred MPITopology{2}(comm_cart)
    @inferred MPITopologies.get_cart_ranks_subcomm(pen1.topology.subcomms[1])

    @inferred PencilArrays.to_local(pen2, (1:2, 1:2, 1:2), MemoryOrder())
    @inferred PencilArrays.to_local(pen2, (1:2, 1:2, 1:2), LogicalOrder())

    @inferred PencilArrays.size_local(pen2, MemoryOrder())

    T = Int
    @inferred PencilArray{T}(undef, pen2)
    @inferred PencilArray{T}(undef, pen2, 3, 4)

    u1 = PencilArray{T}(undef, pen1)
    u2 = similar(u1, pen2)

    @inferred Nothing gather(u2)
    @inferred transpose!(u2, u1)
    @inferred Transpositions.get_remote_indices(1, (2, 3), 8)
end
