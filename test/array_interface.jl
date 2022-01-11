#!/usr/bin/env julia

using MPI
using PencilArrays
using Random
using Test

import ArrayInterface:
    ArrayInterface,
    StaticInt,
    StaticBool,
    contiguous_axis,
    contiguous_axis_indicator,
    contiguous_batch_size,
    dense_dims,
    stride_rank

struct DummyArray{T,N} <: AbstractArray{T,N}
    dims :: Dims{N}
    DummyArray{T}(::UndefInitializer, dims) where {T} = new{T,length(dims)}(dims)
end

Base.size(x::DummyArray) = x.dims
Base.getindex(::DummyArray{T}, ind...) where {T} = zero(T)
Base.strides(x::DummyArray) = Base.size_to_strides(1, size(x)...)

function non_dense_array(::Type{T}, dims) where {T}
    # Only the first dimension is dense: (True, False, False, ...).
    N = length(dims)
    dims_parent = ntuple(d -> (d - 1) + dims[d], Val(N))
    up = view(Array{T}(undef, dims_parent), Base.OneTo.(dims)...)
    @assert dense_dims(up) === ntuple(d -> StaticBool(d == 1), Val(ndims(up)))
    up
end

function non_contiguous_array(::Type{T}, dims) where {T}
    N = length(dims)
    dims_parent = (2, dims...)  # we take the slice [1, :, :, ...]
    up = view(Array{T}(undef, dims_parent), 1, ntuple(d -> Colon(), Val(N))...)
    @assert contiguous_axis(up) === StaticInt(-1)
    @assert size(up) == dims
    @assert ArrayInterface.size(up) == dims
    up
end

function test_array_interface(p::Pencil)
    # Test different kinds of parent arrays
    dims_mem = size_local(p, MemoryOrder())
    up_regular = Array{Float64}(undef, dims_mem)
    up_noncontig = non_contiguous_array(Float64, dims_mem)
    up_nondense = non_dense_array(Float64, dims_mem)
    up_dummy = DummyArray{Float64}(undef, dims_mem)

    parents = (
        up_regular,
        up_noncontig,
        up_nondense,
        up_dummy,
    )

    @testset "Parent $(typeof(up))" for up in parents
        u = PencilArray(p, up)

        @test ArrayInterface.parent_type(u) === typeof(up)
        @test ArrayInterface.known_length(u) === missing
        @test !ArrayInterface.can_change_size(u)
        @test ArrayInterface.ismutable(u)
        @test ArrayInterface.can_setindex(u)
        @test ArrayInterface.aos_to_soa(u) === u
        @test ArrayInterface.fast_scalar_indexing(u)
        @test !ArrayInterface.isstructured(u)

        # Compare outputs with equivalent PermutedDimsArray
        iperm = inv(permutation(u))
        vp = PermutedDimsArray(up, iperm)

        functions = (
            contiguous_axis, contiguous_axis_indicator,
            contiguous_batch_size, stride_rank, dense_dims,
            ArrayInterface.size, ArrayInterface.strides, ArrayInterface.offsets,
        )

        for f in functions
            @inferred f(u)
            @test f(u) === f(vp)
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

topo = MPITopology(comm, Val(2))

pen1 = Pencil(topo, Nxyz, (2, 3))
pen2 = Pencil(pen1, decomp_dims=(1, 3), permute=Permutation(2, 1, 3))
pen3 = Pencil(pen2, decomp_dims=(1, 2), permute=Permutation(3, 2, 1))
pens = (pen1, pen2, pen3)

@testset "ArrayInterface -- Pencil$(decomposition(p))" for p in pens
    test_array_interface(p)
end
