#!/usr/bin/env julia

using MPI
using PencilArrays
using Random
using Test

import PencilArrays:
    inverse_permutation

import ArrayInterface:
    ArrayInterface,
    Contiguous,
    contiguous_axis, contiguous_axis_indicator, contiguous_batch_size,
    stride_rank, DenseDims, dense_dims

struct DummyArray{T,N} <: AbstractArray{T,N}
    dims :: Dims{N}
    DummyArray{T}(::UndefInitializer, dims) where {T} = new{T,length(dims)}(dims)
end

Base.size(x::DummyArray) = x.dims
Base.getindex(::DummyArray{T}, ind...) where {T} = zero(T)

function non_dense_array(::Type{T}, dims) where {T}
    # Only the first dimension is dense: DenseDims((true, false, false, ...)).
    N = length(dims)
    dims_parent = ntuple(d -> (d - 1) + dims[d], Val(N))
    up = view(Array{T}(undef, dims_parent), Base.OneTo.(dims)...)
    @assert dense_dims(up) === DenseDims(ntuple(d -> d == 1, Val(ndims(up))))
    up
end

function test_array_interface(p::Pencil)
    # Test different kinds of parent arrays
    dims_mem = size_local(p, MemoryOrder())
    up_regular = Array{Float64}(undef, dims_mem)
    up_nondense = non_dense_array(Float64, dims_mem)
    up_dummy = DummyArray{Float64}(undef, dims_mem)
    parents = (up_regular, up_nondense, up_dummy)

    @testset "Parent $(typeof(up))" for up in parents
        u = PencilArray(p, up)

        @test ArrayInterface.parent_type(u) === typeof(up)
        @test ArrayInterface.known_length(u) === nothing
        @test !ArrayInterface.can_change_size(u)
        @test ArrayInterface.ismutable(u)
        @test ArrayInterface.can_setindex(u)
        @test ArrayInterface.aos_to_soa(u) === u
        @test ArrayInterface.fast_scalar_indexing(u)
        @test !ArrayInterface.isstructured(u)

        # Compare outputs with equivalent PermutedDimsArray
        iperm = inverse_permutation(get_permutation(u))
        vp = iperm === NoPermutation() ?  up : PermutedDimsArray(up, Tuple(iperm))

        for f in (contiguous_axis, contiguous_batch_size, stride_rank, dense_dims)
            @inferred f(u)
            @test f(u) === f(vp)
        end

        let f = contiguous_axis_indicator
            if contiguous_axis(u) === nothing
                # contiguous_axis_indicator is not defined for this case
                # https://github.com/SciML/ArrayInterface.jl/issues/84
                @test_throws MethodError f(u)
            else
                @inferred f(u)
                @test f(u) === f(vp)
            end
        end
    end

    nothing
end

MPI.Init()
Nxyz = (11, 21, 32)
comm = MPI.COMM_WORLD
Nproc = MPI.Comm_size(comm)
myrank = MPI.Comm_rank(comm)

let dev_null = @static Sys.iswindows() ? "nul" : "/dev/null"
    MPI.Comm_rank(comm) == 0 || redirect_stdout(open(dev_null, "w"))
end

rng = MersenneTwister(42 + myrank)

proc_dims = let pdims = zeros(Int, 2)
    MPI.Dims_create!(Nproc, pdims)
    pdims[1], pdims[2]
end

topo = MPITopology(comm, proc_dims)

pen1 = Pencil(topo, Nxyz, (2, 3))
pen2 = Pencil(pen1, decomp_dims=(1, 3), permute=Permutation(2, 1, 3))
pen3 = Pencil(pen2, decomp_dims=(1, 2), permute=Permutation(3, 2, 1))
pens = (pen1, pen2, pen3)

@testset "ArrayInterface -- Pencil$(get_decomposition(p))" for p in pens
    test_array_interface(p)
end
