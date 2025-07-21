# Test interaction with DifferentialEquations.jl.
# We solve a trivial decoupled system of ODEs.

import DiffEqBase
using MPI
using PencilArrays
using OrdinaryDiffEq
using RecursiveArrayTools: ArrayPartition
using StructArrays: StructArray
using StaticArrays: SVector
using Test

function to_structarray(us::NTuple{N, A}) where {N, A <: AbstractArray}
    T = eltype(A)
    Vec = SVector{N, T}
    StructArray{Vec}(us)
end

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
rank == 0 || redirect_stdout(devnull)

dims = (13, 23, 17)  # prime dimensions to test an unbalanced partitioning
coords_global = map(N -> range(0, 1; length = N), dims)

perm = Permutation(2, 3, 1)
pen = Pencil(dims, comm; permute = perm)
grid = localgrid(pen, coords_global)

u0 = PencilArray{Float64}(undef, pen)
@. u0 = grid.x * grid.y + grid.z

function rhs!(du, u, p, t)
    @. du = -0.1 * u
    du
end

@testset "DiffEqBase" begin
    unorm = DiffEqBase.ODE_DEFAULT_NORM(u0, 0.0)
    unorms = MPI.Allgather(unorm, comm)
    @test allequal(unorms)

    # Note that ODE_DEFAULT_UNSTABLE_CHECK calls NAN_CHECK.
    w = copy(u0)
    wcheck = DiffEqBase.ODE_DEFAULT_UNSTABLE_CHECK(nothing, w, nothing, nothing)
    @test wcheck == false

    # After setting a single value to NaN, all processes should detect it.
    if rank == 0
        w[1] = NaN
    end
    wcheck = DiffEqBase.ODE_DEFAULT_UNSTABLE_CHECK(nothing, w, nothing, nothing)
    @test wcheck == true
end

@testset "OrdinaryDiffEq" begin
    tspan = (0.0, 1000.0)
    params = (;)
    prob = @inferred ODEProblem{true}(rhs!, u0, tspan, params)

    # This is not fully inferred...
    integrator = init(
        prob, Tsit5();
        adaptive = true, save_everystep = false,
    )

    # Check that all timesteps are the same
    for _ = 1:10
        local dts = MPI.Allgather(integrator.dt, comm)
        @test allequal(dts)
        step!(integrator)
    end

    @testset "ArrayPartition" begin
        v0 = ArrayPartition(u0)
        prob = @inferred ODEProblem{true}(rhs!, v0, tspan, params)

        # TODO for now this fails when permutations are enabled due to incompatible
        # broadcasting.
        @test_skip integrator = init(
            prob, Tsit5();
            adaptive = true, save_everystep = false,
        )
    end

    # Solve the equation for a 2D vector field represented by a StructArray.
    @testset "StructArray" begin
        v0 = to_structarray((u0, 2u0))
        @assert eltype(v0) <: SVector{2}
        tspan = (0.0, 1.0)
        prob = @inferred ODEProblem{true}(rhs!, v0, tspan, params)
        integrator = init(
            prob, Tsit5();
            adaptive = true, save_everystep = false,
        )
        @test integrator.u == v0
        for _ ∈ 1:10
            step!(integrator)
        end
        @test integrator.u ≠ v0
    end
end
