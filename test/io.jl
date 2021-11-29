# Packages must be loaded in this order!
using MPI
using HDF5
using PencilArrays
using PencilArrays.PencilIO

import JSON3

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
    root = 0
    rank = MPI.Comm_rank(comm)

    X = (u, u .+ 1, u .+ 2, u .+ 3, u .+ 4)

    kws = Iterators.product((false, true), (false, true))

    @test_nowarn open(MPIIODriver(), filename, comm, write=true, create=true) do ff
        pos = 0
        for (i, (collective, chunks)) in enumerate(kws)
            name = "field_$i"
            ff[name, collective=collective, chunks=chunks] = X[i]
            pos += sizeof_global(X[i])
            @test position(ff) == pos
        end
    end

    # Append some data.
    open(MPIIODriver(), filename, comm, write=true, append=true) do ff
        ff["field_5", chunks=false] = X[5]
        ff["collection"] = X
    end

    @test isfile("$filename.json")
    meta = open(JSON3.read, "$filename.json", "r").datasets

    # Test file contents in serial mode.
    # First, gather data from all processes.
    # Note that we may need to permute indices of data, since data on disk is
    # written in memory (not logical) order.
    perm = Tuple(permutation(u))
    Xg = map(X) do x
        xg = gather(x, root)  # note: data is in logical order
        xg === nothing && return xg
        perm === nothing && return xg
        PermutedDimsArray(xg, perm)
    end

    @test (Xg[1] === nothing) == (rank != root)
    if rank == root
        open(filename, "r") do ff
            y = similar(Xg[1])
            for (i, (collective, chunks)) in enumerate(kws)
                mpiio_read_serial!(ff, y, meta, "field_$i")
                if !chunks  # if chunks = true, data is reordered into blocks
                    @test y == Xg[i]
                end
            end
            # Verify appended data
            mpiio_read_serial!(ff, y, meta, "field_5")
            @test y == Xg[5]
            let y = similar.(Xg)
                mpiio_read_serial!(ff, y, meta, "collection")
                @test all(y .== Xg)
            end
        end
    end

    # Read stuff
    y = similar(X[1])
    @test_nowarn open(MPIIODriver(), filename, comm, read=true) do ff
        @test_throws ErrorException read!(ff, y, "field not in file")
        let y = similar.(X)
            read!(ff, y, "collection")
            @test all(y .== X)
        end
        for (i, (collective, _)) in enumerate(kws)
            name = "field_$i"
            read!(ff, y, name, collective=collective)
            @test y == X[i]
        end
        read!(ff, y, "field_5")
        @test y == X[5]
    end

    nothing
end

read_array!(ff, x) = read!(ff, x)
read_array!(ff, t::Tuple) = map(x -> read_array!(ff, x), t)

function mpiio_read_serial!(ff, x, meta, name)
    offset = meta[Symbol(name)].offset_bytes :: Int
    seek(ff, offset)
    read_array!(ff, x)
end

function test_write_hdf5(filename, u::PencilArray)
    comm = get_comm(u)
    rank = MPI.Comm_rank(comm)
    v = u .+ 1
    w = u .+ 2

    # Open file in serial mode first.
    if rank == 0
        h5open(filename, "w") do ff
            # "HDF5 file was not opened with the MPIO driver"
            @test_throws ErrorException ff["scalar"] = u
        end
    end

    MPI.Barrier(comm)

    @test_nowarn open(PHDF5Driver(), filename, comm, write=true) do ff
        @test isopen(ff)
        @test_nowarn ff["scalar", collective=true, chunks=false] = u
        @test_nowarn ff["vector_tuple", collective=false, chunks=true] = (u, v, w)
        @test_nowarn ff["vector_array", collective=true, chunks=true] = [u, v, w]
    end

    @test_nowarn open(PHDF5Driver(), filename, comm, append=true) do ff
        @test isopen(ff)
        @test_nowarn ff["scalar_again"] = u
    end

    @test_nowarn open(PHDF5Driver(), filename, comm, read=true) do ff
        @test isopen(ff)
        uvw = (u, v, w)
        uvw_r = similar.(uvw)
        ur, vr, wr = uvw_r

        read!(ff, ur, "scalar")
        @test u == ur
        read!(ff, vr, "scalar_again")
        @test vr == ur

        let perm = Tuple(permutation(ur))
            @test haskey(attributes(ff["scalar"]), "permutation")
            expected = perm === nothing ? false : collect(perm)
            @test read(ff["scalar"]["permutation"]) == expected
        end

        read!(ff, uvw_r, "vector_tuple")
        @test all(uvw .== uvw_r)

        fill!.(uvw_r, 0)
        read!(ff, collect(uvw_r), "vector_array", collective=false)
        @test all(uvw .== uvw_r)
    end

    nothing
end

MPI.Init()

Nxyz = (16, 21, 41)
comm = MPI.COMM_WORLD
Nproc = MPI.Comm_size(comm)
myrank = MPI.Comm_rank(comm)

silence_stdout(comm)

@show HDF5.API.libhdf5

@testset "HDF5" begin
    let fapl = HDF5.FileAccessProperties()
        @test PencilIO._is_set(fapl, Val(:fclose_degree)) === false
        fapl.fclose_degree = :strong
        @test PencilIO._is_set(fapl, Val(:fclose_degree)) === true
        @test fapl.fclose_degree === :strong
    end
end

rng = MersenneTwister(42)
perms = (NoPermutation(), Permutation(2, 3, 1))

@testset "$perm" for perm in perms
    pen = Pencil(Nxyz, (1, 3), comm; permute = perm)
    u = PencilArray{Float64}(undef, pen)
    randn!(rng, u)
    u .+= 10 * myrank

    @testset "MPI-IO" begin
        filename = MPI.bcast(tempname(), 0, comm)
        test_write_mpiio(filename, u)
    end

    @testset "HDF5" begin
        filename = MPI.bcast(tempname(), 0, comm)
        test_write_hdf5(filename, u)
    end
end

HDF5.API.h5_close()
MPI.Finalize()
