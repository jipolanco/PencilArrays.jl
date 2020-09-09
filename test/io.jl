#!/usr/bin/env julia

# Packages must be loaded in this order!
using MPI
using HDF5
using PencilArrays
using PencilArrays.PencilIO

if !PencilIO.hdf5_has_parallel()
    @warn "HDF5 has no parallel support. Skipping HDF5 tests."
    exit(0)
end

using Random
using Test

include("include/MPITools.jl")
using .MPITools

function test_write_mpiio(filename, u::PencilArray)
    comm = get_comm(u)

    # TODO simplify this once broadcasting works properly...
    # (example: u .+ 1 should return a PencilArray)
    X = (u, copy(u), copy(u), copy(u))
    X[2] .+= 1
    X[3] .+= 2
    X[4] .+= 3
    
    kws = Iterators.product((false, true), (false, true))

    @test_nowarn open(MPIIODriver(), filename, comm,
                      write=true, create=true) do ff
        off = 0
        for (u, (collective, chunks)) in zip(X, kws)
            off += write(ff, u, offset=off, collective=collective, chunks=chunks)
        end
    end

    # TODO
    # - append data
    # - collections

    # Read stuff
    Y = similar.(X)
    @test_nowarn open(MPIIODriver(), filename, comm, read=true) do ff
        off = 0
        for (u, (collective, chunks)) in zip(Y, kws)
            off += read!(ff, u, offset=off, collective=collective, chunks=chunks)
        end
    end

    for (x, y) in zip(X, Y)
        @test x == y
    end

    nothing
end

function test_write_hdf5(filename, u::PencilArray)
    comm = get_comm(u)
    info = MPI.Info()
    rank = MPI.Comm_rank(comm)

    v = copy(u)
    w = copy(u)
    v .+= 1
    w .+= 2

    # Open file in serial mode first.
    if rank == 0
        h5open(filename, "w") do ff
            # "HDF5 file was not opened with the MPIO driver"
            @test_throws ErrorException ff["scalar"] = u
        end
    end

    MPI.Barrier(comm)

    @test_nowarn ph5open(filename, "w", comm, info) do ff
        @test isopen(ff)
        @test_nowarn ff["scalar", collective=true, chunks=false] = u
        @test_nowarn ff["vector_tuple", collective=false, chunks=true] = (u, v, w)
        @test_nowarn ff["vector_array", collective=true, chunks=true] = [u, v, w]
    end

    @test_nowarn open(PHDF5Driver(), filename, comm, append=true) do ff
        @test isopen(ff)
        @test_nowarn ff["scalar_again"] = u
    end

    @test_nowarn ph5open(filename, "r", comm, info) do ff
        @test isopen(ff)
        uvw = (u, v, w)
        uvw_r = similar.(uvw)
        ur, vr, wr = uvw_r

        read!(ff, ur, "scalar")
        @test u == ur
        read!(ff, vr, "scalar_again")
        @test vr == ur

        read!(ff, uvw_r, "vector_tuple")
        @test all(uvw .== uvw_r)

        fill!.(uvw_r, 0)
        read!(ff, collect(uvw_r), "vector_array", collective=false)
        @test all(uvw .== uvw_r)
    end

    @test_nowarn open(PHDF5Driver(), filename, comm, read=true) do ff
        @test isopen(ff)
    end

    nothing
end

function main()
    MPI.Init()

    Nxyz = (16, 21, 41)
    comm = MPI.COMM_WORLD
    Nproc = MPI.Comm_size(comm)
    myrank = MPI.Comm_rank(comm)

    silence_stdout(comm)

    # Let MPI_Dims_create choose the values of (P1, P2).
    proc_dims = let pdims = zeros(Int, 2)
        MPI.Dims_create!(Nproc, pdims)
        pdims[1], pdims[2]
    end

    rng = MersenneTwister(42)

    topo = MPITopology(comm, proc_dims)
    pen = Pencil(topo, Nxyz, (1, 3), permute=Permutation(2, 3, 1))
    u = PencilArray{Float64}(undef, pen)
    randn!(rng, u)
    u .+= 10 * myrank

    filename = MPI.bcast(tempname(), 0, comm)

    @testset "Write MPI-IO" begin
        test_write_mpiio(filename, u)
    end

    @testset "Write HDF5" begin
        test_write_hdf5(filename, u)
    end

    HDF5.h5_close()
    MPI.Finalize()
end

main()
