#!/usr/bin/env julia

using MPI
using PencilArrays
using Random
using Test

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
Nproc = MPI.Comm_size(comm)

topo = MPITopology(comm, (Nproc, ))

pen = Pencil(topo, (11, 12), (2, ))

A = PencilArray{Float64}(undef, pen)
G = global_view(A)
randn!(A)

function test_broadcast(A)
    @test typeof(2A) == typeof(A)
    @test typeof(A .+ A) == typeof(A)
    @test parent(2A) == 2parent(A)
    nothing
end

@testset "Broadcast $(typeof(x))" for x in (A, G)
    test_broadcast(x)
end

@testset "Combinations" begin
    # Combine with regular Array
    P = parent(A) :: Array
    @test typeof(P .+ A) == typeof(A)

    # Combine PencilArray and GlobalPencilArray
    @test_throws ArgumentError A .+ G
    # @test_throws ArgumentError P .+ G  # this is still allowed...
end

# TODO
# - global and local indexing: make them incompatible?
# - in-place broadcasting?
#   * check that it doesn't allocate
# - permutations?
#   * performance / iteration order?
#   * combinations with non-permuted arrays?
