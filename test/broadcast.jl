using MPI
using PencilArrays
using Random
using Test

using GPUArrays
include("include/jlarray.jl")
using .JLArrays

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
    randn!(A)
    perm = permutation(A)

    @testset "Broadcast" begin
        @test typeof(2A) == typeof(A)
        @test typeof(A .+ A) == typeof(A)
        @test typeof(A .+ A .+ 3) == typeof(A)
        @test parent(2A) == 2parent(A)
        let x = A, y = similar(x)
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
    end

    @testset "GPU arrays" begin
        pp = Pencil(JLArray, pen)
        u = PencilArray{Float32}(undef, pp)
        randn!(u)

        # Some basic stuff that should work without scalar indexing
        # (Nothing to do with broadcasting though...)
        v = @test_nowarn copy(u)
        @test typeof(v) === typeof(u)
        @test_nowarn v == u
        @test v == u
        @test v ≈ u

        @test parent(u) isa JLArray
        @test_nowarn u .+ u  # should avoid scalar indexing
        @test u .+ u == 2u
        @test typeof(u .+ u) == typeof(u)

        @test_nowarn v .= u .+ 2u
        @test typeof(v) == typeof(u)
        @test parent(v) ≈ 3parent(u)
    end
end
