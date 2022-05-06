using MPI
using PencilArrays
using Random
using Test

MPI.Init()

dims = (8, 12, 9)
comm = MPI.COMM_WORLD
perm = Permutation(2, 3, 1)
pen = Pencil(dims, comm; permute = perm)

MPI.Comm_rank(comm) == 0 || redirect_stdout(devnull)

u = PencilArray{Float32}(undef, pen)
randn!(u)
ug = global_view(u)

@testset "Indices" begin
    for A in (u, ug)
        @test all(zip(LinearIndices(A), CartesianIndices(A))) do (n, I)
            A[n] === A[I]
        end

        @test all(pairs(IndexLinear(), A)) do (n, v)
            A[n] === v
        end

        @test all(pairs(IndexCartesian(), A)) do (I, v)
            A[I] === v
        end
    end
end

@testset "Singleton" begin
    Nxg, Nyg, Nzg = size_global(pen, LogicalOrder())
    Nxl, Nyl, Nzl = size_local(pen, LogicalOrder())

    # Two different ways of doing the same thing.
    ux = @inferred PencilArray{Int}(undef, pen; singleton = 1)
    uy = @inferred similar(ux, Int; singleton = 1)
    @test typeof(ux) === typeof(uy)
    @test size(ux) === size(uy)
    @test size(ux) === size_local(ux, LogicalOrder())
    @test size(ux, 1) == 1  # singleton dimension
    @test length(ux) == Nyl * Nzl
    @test length_global(ux) == Nyg * Nzg
    @test size_local(ux, LogicalOrder()) == (1, Nyl, Nzl)
    @test size_global(ux, LogicalOrder()) == (1, Nyg, Nzg)
    @test size_local(ux, MemoryOrder()) == perm * (1, Nyl, Nzl)
    @test size_global(ux, MemoryOrder()) == perm * (1, Nyg, Nzg)
end
