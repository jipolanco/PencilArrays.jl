using MPI
using PencilArrays
using Random
using Test

# TODO test this?
# using GPUArrays
# include("include/jlarray.jl")
# using .JLArrays

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
rank == 0 || redirect_stdout(devnull)

topo = MPITopology(comm, Val(1))

dims = (11, 12, 2)
perm = Permutation(2, 3, 1)
@assert inv(perm) != perm

pencils = (
    "Non-permuted" => Pencil(topo, dims, (2, )),
    "Permuted" => Pencil(topo, dims, (2, ); permute = perm),
)

@testset "$s" for (s, pen) in pencils
    A = PencilArray{Float64}(undef, pen)
    G = global_view(A)
    randn!(A)
    perm = permutation(A)

    @testset "Broadcast $(nameof(typeof(x)))" for x in (A, G)
        @test typeof(2A) == typeof(A)
        @test typeof(A .+ A) == typeof(A)
        @test typeof(A .+ A .+ 3) == typeof(A)
        @test parent(2A) == 2parent(A)
        let y = similar(x)
            broadcast!(+, y, x, x, 3)  # precompile before measuring allocations
            alloc = @allocated broadcast!(+, y, x, x, 3)
            @test alloc == 0
            @test y ≈ 2x .+ 3
        end
    end

    @testset "Combinations" begin
        # Combine with regular Array
        P = parent(A) :: Array
        @test typeof(P .+ A) == typeof(A)
        @test P .+ A == 2A

        # Combine PencilArray and GlobalPencilArray
        @test_throws ArgumentError A .+ G
        # @test typeof(A .+ G) == typeof(G .+ A) == typeof(A)  # PencilArray wins
        # @test A .+ G == 2A

        # Combine Array and GlobalPencilArray
        @test typeof(P .+ G) == typeof(G) == typeof(2G)
        @test P .+ G == 2G
    end
end
