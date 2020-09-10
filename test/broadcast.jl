#!/usr/bin/env julia

using MPI
using PencilArrays
using Random
using Test

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
Nproc = MPI.Comm_size(comm)

let dev_null = @static Sys.iswindows() ? "nul" : "/dev/null"
    rank == 0 || redirect_stdout(open(dev_null, "w"))
end

function test_pencil(pen)
    A = PencilArray{Float64}(undef, pen)
    G = global_view(A)
    randn!(A)
    perm = Tuple(get_permutation(A))

    @testset "Broadcast $(nameof(typeof(x)))" for x in (A, G)
        test_broadcast(x)
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
        P′ = perm === nothing ? P : PermutedDimsArray(P, perm)
        @test typeof(P′ .+ A) == typeof(A)

        # Combine PencilArray and GlobalPencilArray
        @test_throws ArgumentError A .+ G

        # Combine Array and GlobalPencilArray
        @test_throws ArgumentError P .+ G
    end
end

function test_broadcast(A)
    @test typeof(2A) == typeof(A)
    @test typeof(A .+ A) == typeof(A)
    @test typeof(A .+ A .+ 3) == typeof(A)
    @test parent(2A) == 2parent(A)
    nothing
end

topo = MPITopology(comm, (Nproc, ))

pencils = (
    "Non-permuted" => Pencil(topo, (11, 12), (2, )),
    "Permuted" => Pencil(topo, (11, 12), (2, ), permute=Permutation(2, 1)),
)

@testset "$s" for (s, pen) in pencils
    test_pencil(pen)
end
