using PencilArrays
using MPI
using Test
using Random

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

function test_transpose(method)
    dims = (16, 21, 41)
    comm = MPI.COMM_WORLD

    pen1 = @inferred Pencil(dims, (2, 3), comm)
    pen2 = @inferred Pencil(pen1; decomp_dims = (1, 3), permute = Permutation(2, 3, 1))
    pen3 = @inferred Pencil(pen2; decomp_dims = (1, 2), permute = Permutation(3, 2, 1))

    T = Float64
    u1 = PencilArray{T}(undef, pen1)
    u2 = PencilArray{T}(undef, pen2)
    u3 = PencilArray{T}(undef, pen3)

    # Set initial random data.
    myrank = MPI.Comm_rank(comm)
    rng = MersenneTwister(42 + myrank)
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

    # Test transpositions with unsorted decomp_dims (#57).
    let pen_alt = @inferred Pencil(pen1, decomp_dims = (2, 1))
        ualt = PencilArray{T}(undef, pen_alt)
        transpose!(ualt, u1, method=method)
        @test compare_distributed_arrays(u1, ualt)
    end
end

MPI.Init()

comm = MPI.COMM_WORLD
MPI.Comm_rank(comm) == 0 || redirect_stdout(devnull)

transpose_methods = (
    Transpositions.PointToPoint(),
    Transpositions.Alltoallv(),
)

@testset "transpose! $method" for method in transpose_methods
    test_transpose(method)
end
