# Test interaction with DifferentialEquations.jl.
# We solve a trivial decoupled system of ODEs.

using MPI
using PencilArrays
using OrdinaryDiffEq
using RecursiveArrayTools: ArrayPartition
using Test

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
    @. du = 0.1 * u
    du
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
        @test all(==(dts[1]), dts)  # = allequal(dts) on Julia ≥ 1.8
        step!(integrator)
    end

    @testset "ArrayPartition" begin
        v0 = RAT.ArrayPartition(u0)
        prob = @inferred ODEProblem{true}(rhs!, v0, tspan, params)

        # TODO for now this fails when permutations are enabled due to incompatible
        # broadcasting.
        @test_skip integrator = init(
            prob, Tsit5();
            adaptive = true, save_everystep = false,
        )
    end
end
